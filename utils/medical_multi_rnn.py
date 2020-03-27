import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.autograd import Variable
import torchvision.models as models
from torch.nn import init
import torch.nn.functional as F


class VisualFeatureExtractor(nn.Module):
    def __init__(self, hidden_size, embed_size):
        super(VisualFeatureExtractor, self).__init__()

        #resnet152 frontal
        resnet_frontal = models.resnet152()
        modules_frontal =  list(resnet_frontal.children())[:-2]
        resnet_conv_frontal = nn.Sequential(*modules_frontal)

        # lateral
        resnet_lateral = models.resnet152()
        modules_lateral = list(resnet_lateral.children())[:-2]
        resnet_conv_lateral = nn.Sequential(*modules_lateral)

        self.avgpool_fun = nn.AvgPool2d( 10 ) # input=(320,320) last conv is 2048*10*10
        self.resnet_conv_frontal = resnet_conv_frontal
        
        self.resnet_conv_lateral =resnet_conv_lateral
        self.affine_lateral_a = nn.Linear(2048, hidden_size) #  v_i = W_a * V
        self.affine_lateral_b = nn.Linear(2048, embed_size)  # v_g =W_b * a^g

        self.affine_frontal_a = nn.Linear(2048, hidden_size) # v_i = W_a * V
        self.affine_frontal_b = nn.Linear(2048, embed_size)  # v_g = W_b * a^g

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        init.kaiming_uniform( self.affine_frontal_a.weight, mode='fan_in' )
        self.affine_frontal_a.bias.data.fill_( 0 )
        init.kaiming_uniform( self.affine_frontal_b.weight, mode='fan_in')
        self.affine_frontal_b.bias.data.fill_( 0 )

        init.kaiming_uniform( self.affine_lateral_a.weight, mode='fan_in')
        self.affine_lateral_a.bias.data.fill_( 0 )
        init.kaiming_uniform( self.affine_lateral_b.weight, mode='fan_in')
        self.affine_lateral_b.bias.data.fill_( 0 )

    def forward(self, image_frontal, image_lateral):
        """
        inputs: image_frontal, image_lateral
        outputs: V_frontal, v_g_frontal,  V_lateral, v_g_lateral
        """
    
        A_frontal = self.resnet_conv_frontal( image_frontal ) # batch_size x 2048x7x7
        # a^g
        a_g_frontal = self.avgpool_fun( A_frontal ).squeeze()  # batch_size x 2048
        #V=[v1, ... v49]
        V_frontal = A_frontal.view(A_frontal.size(0), A_frontal.size(1), -1).transpose(1, 2)
        V_frontal = self.relu( self.affine_frontal_a( self.dropout( V_frontal )))
        v_g_frontal = self.relu( self.affine_frontal_b( self.dropout( a_g_frontal)))

        
        A_lateral = self.resnet_conv_lateral( image_lateral ) # batch_size x 2048 x7 x7
        #a^g
        a_g_lateral =self.avgpool_fun( A_lateral ).squeeze()
        V_lateral = A_lateral.view( A_lateral.size(0), A_lateral.size(1), -1).transpose(1, 2)
        V_lateral = self.relu( self.affine_lateral_a( self.dropout( V_lateral )))
        v_g_lateral = self.relu( self.affine_lateral_b( self.dropout( a_g_lateral )))

        return V_frontal, v_g_frontal, V_lateral, v_g_lateral



class Multi_RNNAttModel(nn.Module):
    def __init__(self, opt):
        super(Multi_RNNAttModel, self).__init__()
        self.vocab_size = 2070
        self.input_encoding_size = 512 #512
        self.rnn_size = 512 #512
        self.drop_prob_lm = 0.5

        self.att_hid_size =  512
        self.S_max = 8
        self.N_max = 50


        self.embed = nn.Embedding(self.vocab_size, self.input_encoding_size)
        
        self.logit = nn.Linear(self.rnn_size, self.vocab_size )  # mlp
        self.ctx2att_frontal = nn.Linear(self.rnn_size, self.att_hid_size)  # W_v * V 
        self.ctx2att_lateral = nn.Linear(self.rnn_size, self.att_hid_size)  # W_v * V 

        self.WordRNN = nn.LSTMCell(self.rnn_size, self.rnn_size)

        self.sentence_encode1 = nn.Conv1d(self.input_encoding_size, 512, 3, 1)
        self.maxpool1d1 = nn.MaxPool1d(self.N_max-2, 100)
        self.sentence_encode2 = nn.Conv1d(512, 512, 3, 1)
        self.maxpool1d2 = nn.MaxPool1d(self.N_max - 4, 100)
        self.sentence_encode3 = nn.Conv1d(512, 512, 3, 1)
        self.maxpool1d3 = nn.MaxPool1d(self.N_max -6, 100)
        self.init_h0 = nn.Linear(512*2, 512)
        self.init_c0 = nn.Linear(512*2, 512)
        self.init_h1 = nn.Linear(512*2, 512)
        self.init_c1 = nn.Linear(512*2, 512)
        self.init_h2 = nn.Linear(512*2, 512)
        self.init_c2 = nn.Linear(512*2, 512)
        self.init_h3 = nn.Linear(512*2, 512)
        self.init_c3 = nn.Linear(512*2, 512)  
        self.init_h4 = nn.Linear(512*2, 512)
        self.init_c4 = nn.Linear(512*2, 512)  
        self.init_h5 = nn.Linear(512*2, 512)
        self.init_c5 = nn.Linear(512*2, 512)  
        self.init_h6 = nn.Linear(512*2, 512)
        self.init_c6 = nn.Linear(512*2, 512)  
        self.init_h7 = nn.Linear(512*2, 512)
        self.init_c7 = nn.Linear(512*2, 512)  
                      
        self.h2att_frontal = nn.Linear(self.rnn_size*3, self.att_hid_size)
        self.h2att_lateral = nn.Linear(self.rnn_size*3, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.init_h = [self.init_h0, self.init_h1, self.init_h2, self.init_h3, self.init_h4, self.init_h5, self.init_h6, self.init_h7]
        self.init_c = [self.init_c0, self.init_c1, self.init_c2, self.init_c3, self.init_c4, self.init_c5, self.init_c6, self.init_c7]
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.__init_weights()


    def __init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.logit.weight.data.uniform_(-0.1, 0.1)
        self.logit.bias.data.fill_(0)
       
        self.ctx2att_frontal.weight.data.uniform_(-0.1, 0.1)
        self.ctx2att_frontal.bias.data.fill_(0)
        self.ctx2att_lateral.weight.data.uniform_(-0.1, 0.1)
        self.ctx2att_lateral.bias.data.fill_(0)
        self.init_h0.weight.data.uniform_(-0.1, 0.1)
        self.init_h0.bias.data.fill_(0)
        self.init_c0.weight.data.uniform_(-0.1, 0.1)
        self.init_c0.bias.data.fill_(0)
        self.init_h1.weight.data.uniform_(-0.1, 0.1)
        self.init_h1.bias.data.fill_(0)
        self.init_c1.weight.data.uniform_(-0.1, 0.1)
        self.init_c1.bias.data.fill_(0)
        self.init_h2.weight.data.uniform_(-0.1, 0.1)
        self.init_h2.bias.data.fill_(0)
        self.init_c2.weight.data.uniform_(-0.1, 0.1)
        self.init_c2.bias.data.fill_(0)
        self.init_h3.weight.data.uniform_(-0.1, 0.1)
        self.init_h3.bias.data.fill_(0)
        self.init_c3.weight.data.uniform_(-0.1, 0.1)
        self.init_c3.bias.data.fill_(0)
        self.init_h4.weight.data.uniform_(-0.1, 0.1)
        self.init_h4.bias.data.fill_(0)
        self.init_c4.weight.data.uniform_(-0.1, 0.1)
        self.init_c4.bias.data.fill_(0)
        self.init_h5.weight.data.uniform_(-0.1, 0.1)
        self.init_h5.bias.data.fill_(0)
        self.init_c5.weight.data.uniform_(-0.1, 0.1)
        self.init_c5.bias.data.fill_(0)
        self.init_h6.weight.data.uniform_(-0.1, 0.1)
        self.init_h6.bias.data.fill_(0)
        self.init_c6.weight.data.uniform_(-0.1, 0.1)
        self.init_c6.bias.data.fill_(0)
        self.init_h7.weight.data.uniform_(-0.1, 0.1)
        self.init_h7.bias.data.fill_(0)
        self.init_c7.weight.data.uniform_(-0.1, 0.1)
        self.init_c7.bias.data.fill_(0)
        self.h2att_frontal.weight.data.uniform_(-0.1, 0.1)
        self.h2att_frontal.bias.data.fill_(0)
        self.h2att_lateral.weight.data.uniform_(-0.1, 0.1)
        self.h2att_lateral.bias.data.fill_(0)
        self.alpha_net.weight.data.uniform_(-0.1, 0.1)
        self.alpha_net.bias.data.fill_(0)
        # self.affine_betaq.weight.data.uniform_(-0.1, 0.1)
        # self.affine_betaq.bias.data.fill_(0)

    def lateral_attention(self, h, att_feats, p_att_feats):#, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att_lateral(h)                               # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = F.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size
        
        weight = F.softmax(dot, dim=1)                             # batch * att_size
        # if att_masks is not None:
        #     weight = weight * att_masks.view(-1, att_size).float()
        #     weight = weight / weight.sum(1, keepdim=True) # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        return att_res

    def frontal_attention(self, h, att_feats, p_att_feats):#, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att_frontal(h)                               # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = F.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size
        
        weight = F.softmax(dot, dim=1)                             # batch * att_size
    
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        return att_res


    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros( bsz, self.rnn_size),
                weight.new_zeros( bsz, self.rnn_size))


    def sentence_encode(self, word_embeddings):
        word_embeddings = word_embeddings.transpose(2, 1)
        out1 = self.relu(self.sentence_encode1(word_embeddings))
        sent_feature1 = self.maxpool1d1(out1).squeeze(2)
        out2 = self.relu(self.sentence_encode2(out1))
        sent_feature2 = self.maxpool1d2(out2).squeeze(2)
        out3 = self.relu(self.sentence_encode3(out2))
        sent_feature3 = self.maxpool1d3(out3).squeeze(2)
        sentence_vector = torch.cat([sent_feature1, sent_feature2, sent_feature3], 1)
        return sentence_vector
        

    def forward(self, fc_feats_frontal, att_feats_frontal, fc_feats_lateral, att_feats_lateral, im2p_captions):
        batch_size = fc_feats_frontal.size(0)
    
        im2p_outputs = fc_feats_frontal.new_zeros(batch_size, im2p_captions.size(1), im2p_captions.size(2)-1, self.vocab_size)
        p_att_feats_frontal = self.ctx2att_frontal(att_feats_frontal)  #is the first item: W*V in (W*V+ W*h)
        p_att_feats_lateral = self.ctx2att_lateral(att_feats_lateral) 

        # concat frontal and lateral
        fc_feats = torch.cat([fc_feats_frontal, fc_feats_lateral], 1) # (batch_size, hidden*2)
        # generate first sentences
        for sent_id in range(1):
            # init wordrnn
            init_h = self.tanh(self.init_h[sent_id](fc_feats))
            init_c = self.tanh(self.init_c[sent_id](fc_feats))
            wordrnn_last_state = (init_h, init_c)
            # wordrnn_last_state = self.init_hidden(batch_size)
            #generate word by word
            for j in range(im2p_captions.size(2)-1):
                it = im2p_captions[:, sent_id, j].clone()
                # break if all the sequences end
                if j >= 1 and im2p_captions[:, sent_id, j].sum() == 0:
                    break
                xt = self.embed(it)
                #xt = torch.cat([xt, fc_feats], 1) # batch_size, 512*2
                xt = F.dropout(self.relu(xt), self.drop_prob_lm, self.training)
                word_h, word_c = self.WordRNN(xt, wordrnn_last_state)
                wordrnn_last_state = (word_h, word_c)
                output = F.dropout(self.logit(word_h), self.drop_prob_lm, self.training) # batch_size, vocab_size
                #im2p_outputs[:, sent_id, j] = F.log_softmax(output, dim=1)
                im2p_outputs[:, sent_id, j] = output

        # generate the next sentence to last 
        for sent_id in range(1, im2p_captions.size(1)):
            # get sentence features
            word_embedding = self.embed(im2p_captions[:, sent_id-1, :]) # (batch_size, max_seq_length, embed_size)
            sentence_vector = self.sentence_encode(word_embedding) #
            v_att_frontal = self.frontal_attention(sentence_vector, att_feats_frontal, p_att_feats_frontal)
            v_att_lateral = self.lateral_attention(sentence_vector, att_feats_lateral, p_att_feats_lateral)

            v_att = torch.cat([v_att_frontal, v_att_lateral] ,1)
            init_h = self.tanh(self.init_h[sent_id](v_att))
            init_c = self.tanh(self.init_c[sent_id](v_att))
            wordrnn_last_state = (init_h, init_c)
            #wordrnn_last_state = self.init_hidden(batch_size)
            # generate word by word
            for j in range(im2p_captions.size(2)-1):
                it = im2p_captions[:, sent_id, j].clone()
                # break if all the sequences end
                if j >= 1 and im2p_captions[:, sent_id, j].sum() == 0:
                    break
                xt = self.embed(it)
                #xt = torch.cat([xt, v_att], 1) # batch_size, hidden*2
                xt = F.dropout(self.relu(xt), self.drop_prob_lm, self.training)
                word_h, word_c = self.WordRNN(xt, wordrnn_last_state)
                wordrnn_last_state = (word_h, word_c)
                output = F.dropout(self.logit(word_h), self.drop_prob_lm, self.training) # batch_size, vocab_size
                #im2p_outputs[:, sent_id, j] = F.log_softmax(output, dim=1)
                im2p_outputs[:, sent_id, j] = output

        return im2p_outputs

    #def sample(self, fc_feats, att_feats, ix_to_word, att_masks=None, opt={} ):
    def sample(self, fc_feats_frontal, att_feats_frontal, fc_feats_lateral, att_feats_lateral, opt={} ):

        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        #if beam_size > 1:
        #    return self._sample_beam(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats_frontal.size(0)
           
        p_att_feats_frontal = self.ctx2att_frontal(att_feats_frontal)  #is the first item: W*V in (W*V+ W*h)
        p_att_feats_lateral = self.ctx2att_lateral(att_feats_lateral) 
   
        # im2p_outputs = []
        im2p_seq = fc_feats_frontal.new_zeros((batch_size, self.S_max, self.N_max), dtype=torch.long)
        im2p_seqLogProbs = fc_feats_frontal.new_zeros(batch_size, self.S_max, self.N_max)

        # concat frontal and lateral
        fc_feats = torch.cat([fc_feats_frontal, fc_feats_lateral], 1) # (batch_size, hidden*2)
        # generate first sentences
        for sent_id in range(1):
            # init wordrnn
            init_h = self.tanh(self.init_h[sent_id](fc_feats))
            init_c = self.tanh(self.init_c[sent_id](fc_feats))
            wordrnn_last_state = (init_h, init_c)
            #wordrnn_last_state = self.init_hidden(batch_size)
            #generate word by word
            for j in range(self.N_max-1):
                if j == 0: # start_token
                    it = fc_feats.new_ones(batch_size, dtype=torch.long) #start tokens :<S>
                elif sample_max: # True
                    im2p_sampleLogprobs, it = torch.max(im2p_logprobs.data, 1)
                else:
                    if temperature == 1.0:
                        prob_prev = torch.exp(im2p_logprobs.data)
                    else:
                        prob_prev = torch.exp(torch.div(im2p_logprobs.data, temperature))
                    it = torch.multinomial(prob_prev, 1)
                    sampleLogprobs = im2p_logprobs.gather(1, it)
                    it = it.view(-1).long()               
                

                if j >= 1: # first generate word
                    if j ==1 :
                        unfinisehd = it > 2
                    else:
                        unfinisehd = unfinisehd * (it > 2)
                    if unfinisehd.sum() == 0:
                        break
                    it =it * unfinisehd.type_as(it)

                    im2p_seq[:, sent_id, j-1] = it
                    im2p_seqLogProbs[:, sent_id, j-1] = im2p_sampleLogprobs.view(-1)
                xt = self.embed(it)
                #xt = torch.cat([xt, fc_feats], 1)
                xt = F.dropout(self.relu(xt), self.drop_prob_lm, self.training)
                #word_rnn_input = torch.cat([xt, v_att], 1)
                word_h, word_c = self.WordRNN(xt, wordrnn_last_state)
                wordrnn_last_state = (word_h, word_c)
                im2p_logprobs = F.log_softmax(self.logit(word_h), 1)


        # generate the next sentence to last 
        for sent_id in range(1, self.S_max):
            # get sentence features
            word_embedding = self.embed(im2p_seq[:, sent_id-1, :]) # (batch_size, max_seq_length, embed_size)
            sentence_vector = self.sentence_encode(word_embedding) #
            #v_att = self.attention(sentence_vector, att_feats, p_att_feats, att_masks)
            v_att_frontal = self.frontal_attention(sentence_vector, att_feats_frontal, p_att_feats_frontal)
            v_att_lateral = self.lateral_attention(sentence_vector, att_feats_lateral, p_att_feats_lateral)

            v_att = torch.cat([v_att_frontal, v_att_lateral] ,1)

            init_h = self.tanh(self.init_h[sent_id](v_att))
            init_c = self.tanh(self.init_c[sent_id](v_att))
            wordrnn_last_state = (init_h, init_c)
            #wordrnn_last_state = self.init_hidden(batch_size)

            # generate word by word
            for j in range(self.N_max-1):
                if j == 0: # start_token
                    it = fc_feats.new_ones(batch_size, dtype=torch.long) #start tokens :<S>
                elif sample_max: # True
                    im2p_sampleLogprobs, it = torch.max(im2p_logprobs.data, 1)
                else:
                    if temperature == 1.0:
                        prob_prev = torch.exp(im2p_logprobs.data)
                    else:
                        prob_prev = torch.exp(torch.div(im2p_logprobs.data, temperature))
                    it = torch.multinomial(prob_prev, 1)
                    sampleLogprobs = im2p_logprobs.gather(1, it)
                    it = it.view(-1).long()               
                

                if j >= 1: # first generate word
                    if j ==1 :
                        unfinisehd = it > 2
                    else:
                        unfinisehd = unfinisehd * (it > 2)
                    if unfinisehd.sum() == 0:
                        break
                    it =it * unfinisehd.type_as(it)

                    im2p_seq[:, sent_id, j-1] = it
                    im2p_seqLogProbs[:, sent_id, j-1] = im2p_sampleLogprobs.view(-1)
                xt = self.embed(it)
                #xt = torch.cat([xt, v_att], 1)
                xt = F.dropout(self.relu(xt), self.drop_prob_lm, self.training)
                #word_rnn_input = torch.cat([xt, v_att], 1)
                word_h, word_c = self.WordRNN(xt, wordrnn_last_state)
                wordrnn_last_state = (word_h, word_c)
                im2p_logprobs = F.log_softmax(self.logit(word_h), 1)


        return im2p_seq





class Attention_lateral(nn.Module):
    def __init__(self, opt):
        super(Attention_lateral, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size*3, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats):#, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att(h)                               # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = F.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size
        
        weight = F.softmax(dot, dim=1)                             # batch * att_size
        # if att_masks is not None:
        #     weight = weight * att_masks.view(-1, att_size).float()
        #     weight = weight / weight.sum(1, keepdim=True) # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        return att_res


class vectorCore(nn.Module):
    def __init__(self):
        super(vectorCore, self).__init__()
        
        self.att_fc = nn.Linear(83, 1)
        self.output_fc = nn.Linear(2, 1)
        self.sf_fc = nn.Linear(2048, 512)
        
    def forward(self, att_feats, fc_feats):
        
        att_feats_reshaped = self.att_fc(att_feats)
        att_feats_reshaped = att_feats_reshaped.squeeze(-1)
	
        stack = torch.stack([att_feats_reshaped, fc_feats], dim=2)
        
        output = self.output_fc(stack)
        output = output.squeeze(-1)
        
        semantic_features = self.sf_fc(output)
        
        return semantic_features

