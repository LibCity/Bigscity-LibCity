import torch
import torch.nn as nn
import torch.nn.functional as F

from trafficdl.model.abstract_model import AbstractModel

class RNN(AbstractModel):

    def __init__(self, config, data_feature):
        super(RNN, self).__init__(config, data_feature)
        self.loc_size = data_feature['loc_size']
        self.loc_emb_size = config['loc_emb_size']
        self.tim_size = data_feature['tim_size']
        self.tim_emb_size = config['tim_emb_size']
        self.hidden_size = config['hidden_size']
        self.gpu = config['gpu']
        self.rnn_type = config['rnn_type']

        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size, padding_idx=data_feature['loc_pad'])
        self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size, padding_idx=data_feature['tim_pad'])
        input_size = self.loc_emb_size + self.tim_emb_size

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, self.hidden_size, 1)
        
        self.init_weights()
        self.fc = nn.Linear(self.hidden_size, self.loc_size)
        self.dropout = nn.Dropout(p=config['dropout_p'])

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

    def forward(self, batch):
        loc = batch['current_loc']
        tim = batch['current_tim']
        batch_size = loc.shape[0]
        
        h1 = torch.zeros(1, batch_size, self.hidden_size)
        c1 = torch.zeros(1, batch_size, self.hidden_size)
        if self.gpu:
            h1 = h1.cuda()
            c1 = c1.cuda()

        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        x = torch.cat((loc_emb, tim_emb), 2).permute(1, 0, 2) # change batch * seq * input_size to seq * batch * input_size
        x = self.dropout(x)

        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            out, h1 = self.rnn(x, h1)
        elif self.rnn_type == 'LSTM':
            out, (h1, c1) = self.rnn(x, (h1, c1))
        # out = out.squeeze(1)
        out = out.permute(1, 0, 2)
        out = F.selu(out)
        out = self.dropout(out)

        y = self.fc(out)
        score = F.log_softmax(y, dim=2)  # calculate loss by NLLoss
        # 因为是补齐了的，所以需要找到真正的 score
        loc_len = batch.get_origin_len('current_loc')
        for i in range(score.shape[0]):
            if i == 0:
                true_scores = score[i][loc_len[i] - 1].reshape(1, -1)
            else:
                true_scores = torch.cat((true_scores, score[i][loc_len[i] - 1].reshape(1, -1)), 0)
        return true_scores

    def predict(self, batch):
        return self.forward(batch)
    
    def calculate_loss(self, batch):
        criterion = nn.NLLLoss()
        if self.gpu:
            criterion = criterion.cuda()
        scores = self.forward(batch)
        return criterion(scores, batch['target'])
