import torch
import torch.nn as nn
import numpy as np
from libcity.model.abstract_model import AbstractModel
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class EmbeddingMatrix(nn.Module):  # text_embdeding
    def __init__(self, input_size, output_size, word_vec):
        super(EmbeddingMatrix, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layer = nn.Linear(in_features=self.input_size, out_features=self.output_size, bias=False)
        self.init_weight(word_vec)

    def init_weight(self, word_vec):
        # word_vec为text_embedding初始权重矩阵,从data_feature传入.
        # self.weight output_size*input_size namely length_of_wordvect_glove_pretrained(50)
        # *text_size(the size of dictionary)
        # 按照论文源代码 word_vec = text_size(the size of dictionary)*length_of_wordvect_glove_pretrained
        word_vec = torch.Tensor(word_vec).t()  # 转置
        self.layer.weight = nn.Parameter(word_vec)

    def forward(self, x):  # x:batch*seq*input_size
        # return torch.matmul(x, self.weights)   #batch*seq*text_size * text_size*output_size = batch*seq*output_size
        return self.layer(x)  # batch*seq*output_size


class SERM(AbstractModel):
    def __init__(self, config, data_feature):
        super(SERM, self).__init__(config, data_feature)
        # initialize parameters
        # print(config['dataset_class'])
        self.loc_size = data_feature['loc_size']
        self.loc_emb_size = config['loc_emb_size']
        self.tim_size = data_feature['tim_size']
        self.tim_emb_size = config['tim_emb_size']
        self.user_size = data_feature['uid_size']
        self.user_emb_size = data_feature['loc_size']  # 根据论文
        self.text_size = data_feature['text_size']
        self.text_emb_size = len(data_feature['word_vec'][0])  # 这个受限于 word_vec 的长度
        self.hidden_size = config['hidden_size']
        self.word_one_hot_matrix = np.eye(self.text_size)
        self.device = config['device']
        # Embedding layer
        self.emb_loc = nn.Embedding(num_embeddings=self.loc_size, embedding_dim=self.loc_emb_size,
                                    padding_idx=data_feature['loc_pad'])
        self.emb_tim = nn.Embedding(num_embeddings=self.tim_size, embedding_dim=self.tim_emb_size,
                                    padding_idx=data_feature['tim_pad'])
        self.emb_user = nn.Embedding(num_embeddings=self.user_size, embedding_dim=self.user_emb_size)
        self.emb_text = EmbeddingMatrix(self.text_size, self.text_emb_size, data_feature['word_vec'])

        # lstm layer
        self.lstm = nn.LSTM(input_size=self.loc_emb_size + self.tim_emb_size + self.text_emb_size,
                            hidden_size=self.hidden_size)
        # self.lstm = nn.LSTM(input_size=self.loc_emb_size + self.tim_emb_size, hidden_size=self.hidden_size)
        # dense layer
        self.dense = nn.Linear(in_features=self.hidden_size, out_features=self.loc_size)
        # init weight
        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)

    def forward(self, batch):

        loc = batch['current_loc']
        tim = batch['current_tim']
        user = batch['uid']
        text = batch['text']
        max_len = batch['current_loc'].shape[1]
        text_pad = np.zeros((self.text_size))
        # text 现在是 word index 的形式，还需要进行 one_hot encoding
        one_hot_text = []
        for word_index in text:
            one_hot_text_a_slice = []
            for words in word_index:
                if len(words) == 0:
                    one_hot_text_a_slice.append(np.zeros((self.text_size)))
                else:
                    one_hot_text_a_slice.append(np.sum(self.word_one_hot_matrix[words], axis=0) /
                                                len(words))
            # pad
            one_hot_text_a_slice += [text_pad] * (max_len - len(one_hot_text_a_slice))
            one_hot_text.append(np.array(one_hot_text_a_slice))  # batch_size * seq_len * text_size

        one_hot_text = torch.FloatTensor(one_hot_text).to(self.device)
        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        user_emb = self.emb_user(user)
        text_emb = self.emb_text(one_hot_text)

        # change batch*seq*emb_size to seq*batch*emb_size
        x = torch.cat([loc_emb, tim_emb, text_emb], dim=2).permute(1, 0, 2)
        # attrs_latent = torch.cat([loc_emb, tim_emb], dim=2).permute(1, 0, 2)
        # print(attrs_latent.size())
        # pack attrs_latent
        seq_len = batch.get_origin_len('current_loc')
        pack_x = pack_padded_sequence(x, lengths=seq_len, enforce_sorted=False)
        lstm_out, (h_n, c_n) = self.lstm(pack_x)  # seq*batch*hidden_size
        # print(lstm_out.size())
        # unpack
        lstm_out, out_len = pad_packed_sequence(lstm_out, batch_first=True)
        # user_emb is batch*loc_size, so we need get the final lstm_out
        for i in range(lstm_out.shape[0]):
            if i == 0:
                out = lstm_out[0][seq_len[i] - 1].reshape(1, -1)  # .reshape(1,-1)表示：转化为1行
            else:
                out = torch.cat((out, lstm_out[i][seq_len[i] - 1].reshape(1, -1)), 0)
        dense = self.dense(out)  # batch * loc_size

        out_vec = torch.add(dense, user_emb)  # batch * loc_size
        pred = nn.LogSoftmax(dim=1)(out_vec)  # result
        # print(pred.size())

        return pred  # batch*loc_size

    def predict(self, batch):
        return self.forward(batch)

    def calculate_loss(self, batch):
        criterion = nn.NLLLoss()
        scores = self.forward(batch)  # batch*loc_size
        return criterion(scores, batch['target'])
