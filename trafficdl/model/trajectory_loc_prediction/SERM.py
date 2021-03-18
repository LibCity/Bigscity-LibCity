import torch
import torch.nn as nn
from trafficdl.model.abstract_model import AbstractModel


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
        self.text_emb_size = config['text_emb_size']
        self.hidden_size = config['hidden_size']

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

        # dense layer
        self.dense = nn.Linear(in_features=self.hidden_size, out_features=self.loc_size)

    def forward(self, batch):
        # input
        # batch*seq
        # user batch
        loc = batch['current_loc']
        tim = batch['current_tim']
        user = batch['uid']
        text = batch['text']

        # print(self.loc_size)
        # print(self.loc_emb_size)
        # print(self.tim_size)
        # print(self.tim_emb_size)
        # print(self.user_size)
        # print(self.user_emb_size)
        # print(self.text_size)
        # print(self.text_emb_size)
        # print(self.hidden_size)

        # print(loc.size())
        # print(tim.size())
        # print(user.size())
        # print(text.size())

        # batch*seq*emb_size
        # user batch*emb_size
        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        user_emb = self.emb_user(user)
        text_emb = self.emb_text(text)

        # print(loc_emb.size())
        # print(tim_emb.size())
        # print(user_emb.size())
        # print(text_emb.size())
        # change batch*seq*emb_size to seq*batch*emb_size
        attrs_latent = torch.cat([loc_emb, tim_emb, text_emb], dim=2).permute(1, 0, 2)
        # print(attrs_latent.size())

        lstm_out, (h_n, c_n) = self.lstm(attrs_latent)  # seq*batch*hidden_size
        # print(lstm_out.size())

        dense = self.dense(lstm_out)  # seq*batch*loc_size
        # print(dense.size())
        # get seq*batch*loc_size then change to batch*seq*loc_size
        out_vec = torch.add(dense, user_emb).permute(1, 0, 2)
        # print(out_vec.size())

        pred = nn.Softmax(dim=2)(out_vec)  # result batch*seq*loc_size
        # print(pred.size())

        # 由于预测的是下一跳，所以选择有效轨迹（补齐前）的最后一个位置的预测值作为实际输出值
        loc_len = batch.get_origin_len('current_loc')
        for i in range(pred.shape[0]):
            if i == 0:
                true_pred = pred[0][loc_len[i] - 1].reshape(1, -1)  # .reshape(1,-1)表示：转化为1行
            else:
                true_pred = torch.cat((true_pred, pred[i][loc_len[i] - 1].reshape(1, -1)), 0)

        return true_pred  # batch*loc_size

    def predict(self, batch):
        return self.forward(batch)

    def calculate_loss(self, batch):
        criterion = nn.NLLLoss()  # CrossEntropyLoss: 1)log softmax 2)cross entropy
        scores = self.forward(batch)  # batch*loc_size
        return criterion(scores, batch['target'])
