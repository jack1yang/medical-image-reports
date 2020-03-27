import time
import pickle
import argparse
from tqdm import tqdm
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.autograd import Variable
from pycocoevalcap.eval import calculate_metrics
from utils.medical_multi_rnn import *
from utils.dataset import *
from utils.logger import Logger



class DebuggerBase:
    def __init__(self, args):
        self.args = args

        self.min_train_loss = 10000000000
        self.params = None

        self._init_model_path()
        self.model_dir = self._init_model_dir()
        self.writer = self._init_writer()
        self.train_transform = self._init_train_transform()
        self.val_transform = self._init_val_transform()
        self.vocab = self._init_vocab()
        self.model_state_dict = self._load_mode_state_dict()

        self.train_data_loader = self._init_data_loader(self.args.train_file_list, self.train_transform, self.args.batch_size)
        self.val_data_loader = self._init_data_loader(self.args.val_file_list, self.val_transform, self.args.val_batch_size)

        self.extractor = self._init_visual_extractor()
        self.multi_rnn_model = self._init_multi_rnn_model()


        self.ce_criterion = self._init_ce_criterion()
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        self.logger = self._init_logger()
        self.writer.write("{}\n".format(self.args))
        
    def train(self):
        bleu_1_max = 0
        max_epoch = 0
        Cider_max = 0.0
        for epoch_id in range(self.start_epoch, self.args.epochs):
                      
            print('===epoch_id:{}==='.format(epoch_id))
            train_loss = self._epoch_train(epoch_id)
            val_tag_loss, val_stop_loss, val_word_loss, val_loss = self._epoch_val()
            bleu_1, bleu_2, bleu_3, bleu_4, Cider, Meteor, Rouge = self.generate_hier_with_spatial(epoch_id, Cider_max)
            if bleu_1 > bleu_1_max:
                bleu_1_max = bleu_1
                max_epoch = epoch_id
            if Cider > Cider_max:
                Cider_max = Cider
            print('*******belu_1max:{}, max_epoch:{}'.format(bleu_1_max, max_epoch))

            if self.args.mode == 'train':
                self.scheduler.step(train_loss)
            else:
                self.scheduler.step(val_loss)
            self.writer.write(
                "[{} - Epoch {}] train loss:{} - val_loss:{} - lr:{}  - bleu_1:{}  - \
                bleu_2:{}  - bleu_3:{}  - bleu_4:{}  - cider:{}  - meteor:{}  - rouge:{}\n".format(self._get_now(),
                                                                               epoch_id,
                                                                               train_loss,
                                                                               val_loss,
                                                                               self.optimizer.param_groups[0]['lr'],
                                                                               bleu_1, bleu_2, bleu_3, bleu_4,
                                                                               Cider, Meteor, Rouge))
            self.writer.flush()                                                                   
            self._save_model(epoch_id,
                             val_loss,
                             val_stop_loss,
                             val_word_loss,
                             train_loss)
            self._log(train_loss=train_loss,                    
                      val_stop_loss=val_stop_loss,
                      val_word_loss=val_word_loss,
                      val_loss=val_loss,
                      lr=self.optimizer.param_groups[0]['lr'],
                      epoch=epoch_id)

    def _epoch_train(self):
        raise NotImplementedError

    def _epoch_val(self):
        raise NotImplementedError

    def _init_train_transform(self):
        transform = transforms.Compose([
            transforms.Resize(self.args.resize),
            transforms.RandomCrop(self.args.crop_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def _init_val_transform(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.crop_size, self.args.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def _init_model_dir(self):
        model_dir = os.path.join(self.args.model_path, self.args.saved_model_name)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_dir = os.path.join(model_dir, self._get_now())

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        return model_dir

    def _init_vocab(self):
        with open(self.args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        self.writer.write("Vocab Size:{}\n".format(len(vocab)))

        return vocab

    def _load_mode_state_dict(self):
        self.start_epoch = 0
        try:
            model_state = torch.load(self.args.load_model_path)
            self.start_epoch = model_state['epoch']
            self.writer.write("[Load Model-{} Succeed!]\n".format(self.args.load_model_path))
            self.writer.write("Load From Epoch {}\n".format(model_state['epoch']))
            return model_state
        except Exception as err:
            self.writer.write("[Load Model Failed] {}\n".format(err))
            return None

    def _init_visual_extractor(self):
        model = VisualFeatureExtractor( self.args.hidden_size, self.args.embed_size)

        try:
            model_state_frontal = torch.load( self.args.load_frontal_visual_model_path)
            model.load_state_dict(model_state_frontal, strict= False)

            model_state_lateral = torch.load( self.args.load_lateral_visual_model_path)
            model.load_state_dict(model_state_lateral, strict=False)
            self.writer.write("[Load Visual Extractor Succeed!]\n")
        except Exception as err:
            self.writer.write("[Load Model Failed] {}\n".format(err))



        if not self.args.visual_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())
               
        if self.args.cuda:
            model = model.cuda()

        return model



    def _init_data_loader(self, file_list, transform, batch_size):
        data_loader = get_loader(image_dir=self.args.image_dir,
                                 caption_json=self.args.caption_json,
                                 file_list=file_list,
                                 vocabulary=self.vocab,
                                 transform=transform,
                                 batch_size=batch_size,
                                 s_max=self.args.s_max,
                                 n_max=self.args.n_max,
                                 shuffle=True)
        return data_loader

    @staticmethod
    def _init_ce_criterion():
        return nn.CrossEntropyLoss(size_average=False, reduce=False)

    @staticmethod
    def _init_mse_criterion():
        return nn.MSELoss()

    def _init_optimizer(self):
        return torch.optim.Adam(params=self.params, lr=self.args.learning_rate)

    def _log(self,
             train_loss,
             val_stop_loss,
             val_word_loss,
             val_loss,
             lr,
             epoch):
        info = {
            'train loss': train_loss,
            'val stop loss': val_stop_loss,
            'val word loss': val_word_loss,
            'val loss': val_loss,
            'learning rate': lr
        }

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, epoch + 1)

    def _init_logger(self):
        logger = Logger(os.path.join(self.model_dir, 'logs'))
        return logger

    def _init_writer(self):
        writer = open(os.path.join(self.model_dir, 'logs.txt'), 'w')
        return writer

    def _to_var(self, x, requires_grad=True):
        if self.args.cuda:
            x = x.cuda(0)
        return Variable(x, requires_grad=requires_grad)

    def _get_date(self):
        return str(time.strftime('%Y%m%d', time.gmtime()))

    def _get_now(self):
        return str(time.strftime('%Y%m%d-%H:%M', time.gmtime()))

    def _init_scheduler(self):
        scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=self.args.patience, factor=0.1)
        return scheduler

    def _init_model_path(self):
        if not os.path.exists(self.args.model_path):
            os.makedirs(self.args.model_path)

    def _init_log_path(self):
        if not os.path.exists(self.args.log_path):
            os.makedirs(self.args.log_path)

    def _save_model(self,
                    epoch_id,
                    val_loss,
                    val_stop_loss,
                    val_word_loss,
                    train_loss):
        def save_whole_model(_filename):
            self.writer.write("Saved Model in {}\n".format(_filename))
            torch.save({'extractor': self.extractor.state_dict(),
                        'multi_rnn_model': self.multi_rnn_model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': epoch_id},
                       os.path.join(self.model_dir, "{}".format(_filename)))

        def save_part_model(_filename, value):
            self.writer.write("Saved Model in {}\n".format(_filename))
            torch.save({"model": value},
                       os.path.join(self.model_dir, "{}".format(_filename)))



        if train_loss < self.min_train_loss:
            file_name = "train_best_loss.pth.tar"
            save_whole_model(file_name)
            self.min_train_loss = train_loss




class Debugger(DebuggerBase):
    def _init_(self, args):
        DebuggerBase.__init__(self, args)
        self.args = args

    def _epoch_train(self, epoch_id):
        stop_loss, word_loss, loss =  0, 0, 0
        dp_extractor = torch.nn.DataParallel( self.extractor)#, device_ids = [0])

        dp_extractor.train()
        dp_multi_rnn_model = torch.nn.DataParallel( self.multi_rnn_model)
        dp_multi_rnn_model.train()


        total_step = len(self.train_data_loader)
        for i, (images_frontal, images_lateral, images_name,  captions, prob) in enumerate(self.train_data_loader):
            batch_stop_loss, batch_word_loss, batch_loss = 0, 0, 0
            images_frontal = self._to_var(images_frontal)
            images_lateral =self._to_var(images_lateral)


            V_frontal, v_g_frontal, V_lateral, v_g_lateral = dp_extractor.forward(images_frontal, images_lateral) 
            captions = self._to_var(torch.Tensor(captions).long(), requires_grad=False)


            para_captions = dp_multi_rnn_model(v_g_frontal, V_frontal, v_g_lateral, V_lateral, captions)           
            batch_loss = self.compute_loss(captions[:,:, 1:], para_captions)
                         
            self.optimizer.zero_grad()
            batch_loss.backward()
            if self.args.clip > 0:
                for param in dp_multi_rnn_model.parameters():
                    if torch.is_tensor(param.grad):
                        param.grad.data.clamp_(-self.args.clip, self.args.clip)
            self.optimizer.step()

            loss += batch_loss.data

            # Print log info
            if i % args.log_step == 0:
                print ('Epoch [%d/%d], Step [%d/%d],  CrossEntropy Loss: %.4f' %( epoch_id, args.epochs, 
                                                                                                 i, total_step, 
                                                                                                batch_loss.item() ))
           
       
        return  loss



    def generate_hier_with_spatial(self, epoch_id, Cider_max):
        self.extractor.eval()
        self.multi_rnn_model.eval()

        results = {}


        for k, (frontal_images, lateral_images, image_names, captions, probs) in enumerate(self.val_data_loader):
            frontal_images = self._to_var(frontal_images, requires_grad=False)
            lateral_images = self._to_var(lateral_images, requires_grad=False)


            V_frontal, v_g_frontal, V_lateral, v_g_lateral = self.extractor.forward(frontal_images, lateral_images)

            
            pred_sentences = {}
            real_sentences = {}
            for image_name in image_names:
                image_prefix = image_name[0].split('_')[0]
                pred_sentences[image_prefix] = {}
                real_sentences[image_prefix] = {}


            sampled_captions = self.multi_rnn_model.sample(v_g_frontal, V_frontal, v_g_lateral, V_lateral)
            for i in range(self.args.s_max):

                sampled_ids = sampled_captions[:, i, :]

                for id, array in zip(image_names, sampled_ids):
                    image_prefix = id[0].split('_')[0]
                    pred_sentences[image_prefix][i] = self.__vec2sent(array.cpu().detach().numpy())
            
  
            for id, array in zip(image_names, captions):
                image_prefix = id[0].split('_')[0]
                for i, sent in enumerate(array):
                    real_sentences[image_prefix][i] = self.__vec2sent(sent[1:])

            
            for image_name in image_names:
                id = image_name[0].split('_')[0]
                results[id] = { 
                    'Pred Sent': pred_sentences[id],
                    'Real Sent': real_sentences[id]
                }
        
        datasetGTS = {'annotations': []}
        datasetRES = {'annotations': []}

        for i, image_id in enumerate(results):
            array = []
            for each in results[image_id]['Pred Sent']:
                array.append(results[image_id]['Pred Sent'][each])
            pred_sent = '. '.join(array)

            array = []
            for each in results[image_id]['Real Sent']:
                sent = results[image_id]['Real Sent'][each]
                if len(sent) != 0:
                    array.append(sent)
            real_sent = '. '.join(array)
            datasetGTS['annotations'].append({
                'image_id': i,
                'caption': real_sent
            })
            datasetRES['annotations'].append({
                'image_id': i,
                'caption': pred_sent
            })

        rng = range(len(results))
        
        evaluations = calculate_metrics(rng, datasetGTS, datasetRES)
        print(type(evaluations))
        print(evaluations)
        if evaluations['CIDEr'] > Cider_max:
            self.__save_json(results, epoch_id)
        return evaluations['Bleu_1'], evaluations['Bleu_2'], evaluations['Bleu_3'], evaluations['Bleu_4'],evaluations['CIDEr'], evaluations['METEOR'],evaluations['ROUGE_L']


    def _epoch_val(self):
        tag_loss, stop_loss, word_loss, loss = 0, 0, 0, 0
        return tag_loss, stop_loss, word_loss, loss


    def __save_json(self, result, epoch_id):
        result_path = os.path.join(self.model_dir, self.args.result_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        with open(os.path.join(result_path, '{}.json'.format(epoch_id)), 'w') as f:
            json.dump(result, f)

    def __vec2sent(self, array):
        sampled_caption = []
        for word_id in array:
            word = self.vocab.get_word_by_id(word_id)
            if word == '<start>':
                continue
            if word == '<end>' or word == '<pad>':
                break
            sampled_caption.append(word)
        return ' '.join(sampled_caption)

    def _init_multi_rnn_model(self):
        model = Multi_RNNAttModel(self.args)
        try:
            model_state = torch.load(self.args.multi_rnn_model_path)
            model.load_state_dict(model_state['model'])
            self.writer.write("[Load Multi_RNN Model Succeed!\n")
        except Exception as err:
            self.writer.write("[Load Multi_RNN model Failed {}!]\n".format(err))

        if not self.args.multi_rnn_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())

        if self.args.cuda:
            model = model.cuda()
        return model

    def compute_loss(self, im2p_caption_labels, im2p_caption_ouput):
        S_max = im2p_caption_ouput.size(1)
        N_max = im2p_caption_ouput.size(2)

        batch_word_loss = 0.0

        for i in range(S_max):
            for j in range(N_max):        
                loss = self.ce_criterion(im2p_caption_ouput[:, i, j, :], im2p_caption_labels[:, i, j])
                loss = loss * (im2p_caption_labels[:, i, j] > 0).float()
                loss = loss.mean()
                batch_word_loss += loss

        return batch_word_loss   


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()

    """
    Data Argument
    """
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--mode', type=str, default='train')

    # Path Argument
    parser.add_argument('--vocab_path', type=str, default='vocab.pkl',
                        help='the path for vocabulary object')
    parser.add_argument('--image_dir', type=str, default='NLMCXR_png_pairs',
                        help='the path for images')
    parser.add_argument('--caption_json', type=str, default='data.json',
                        help='path for captions')
    parser.add_argument('--train_file_list', type=str, default='train_split.txt',
                        help='the train array')
    parser.add_argument('--val_file_list', type=str, default='test_split.txt',
                        help='the val array')
    # transforms argument
    parser.add_argument('--resize', type=int, default=320,
                        help='size for resizing images')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    # Load/Save model argument
    parser.add_argument('--model_path', type=str, default='medical_report_generation_results/',
                        help='path for saving trained models')
    parser.add_argument('--load_model_path', type=str, default='',
                        help='The path of loaded model')
    parser.add_argument('--saved_model_name', type=str, default='_epoch_train_multi_rnn',
                        help='The name of saved model')
    parser.add_argument('--result_path', type=str, default='results', help='generated captions saved path')

    """
    Model Argument
    """
    parser.add_argument('--momentum', type=int, default=0.1)
    # VisualFeatureExtractor
    parser.add_argument('--visual_model_name', type=str, default='resnet152',
                        help='CNN model name')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='not using pretrained model when training')
    parser.add_argument('--fine_tune_start_layer', type=int, default=6)
    parser.add_argument('--load_frontal_visual_model_path', type=str, default='', 
                    help='frontal view pretrained model'    )
    parser.add_argument('--load_lateral_visual_model_path', type=str, default='',
                    help='lateral view pretrained model'    )                    
    parser.add_argument('--load_visual_model_path', type=str,
                        default='.')
    parser.add_argument('--visual_trained', action='store_true', default=True,
                        help='Whether train visual extractor or not')
    parser.add_argument('--num_spatial', type=int, default=49, help='num spatial of last cnn features')


    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=512)




    # Word Model
    parser.add_argument('--multi_rnn_model_path', type=str, default='.')
    parser.add_argument('--multi_rnn_trained', action='store_true', default=True)
    """
    Training Argument
    """
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=int, default=0.001)
    parser.add_argument('--epochs', type=int, default=1000)

    parser.add_argument('--clip', type=float, default=0.1,
                        help='gradient clip, -1 means no clip (default: 0.35)')
    parser.add_argument('--s_max', type=int, default=8)
    parser.add_argument('--n_max', type=int, default=50)

    parser.add_argument('--log_step', type=int, default=10)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print(args)
    debugger = Debugger(args)
    debugger.train()
