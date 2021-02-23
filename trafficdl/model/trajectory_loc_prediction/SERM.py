import torch
import torch.nn as nn
from trafficdl.evaluator.abstract_model import AbstractModel

#not include text embedding
class SERM(AbstractModel):
    def __init__(self, config, data_feature):
        super(SERM,self).__init__()
        #initialize parameters
        self.loc_size = data_feature['loc_size']
        self.loc_emb_size = config['loc_emb_size']
        self.tim_size = data_feature['tim_size']
        self.tim_emb_size = config['tim_emb_size']
        self.user_size = data_feature['user_size']
        self.user_emb_size = config['user_emb_size']
        #self.text_size = data_feature['text_size']
        #self.text_emb_size = config['text_emb_size']
        self.hidden_size = config['hidden_size']

        #Embedding layer
        self.emb_loc = nn.Embedding(num_embeddings = self.loc_size, embedding_dim = self.loc_emb_size, padding_idx = data_feature['loc_pad'])
        self.emb_tim = nn.Embedding(num_embeddings = self.tim_size, embedding_dim = self.tim_emb_size, padding_idx = data_feature['tim_pad'])
        self.emb_user = nn.Embedding(num_embeddings = self.user_size, embedding_dim = self.user_emb_size, padding_idx = data_feature['user_pad'])

        #lstm layer
        self.lstm = nn.LSTM(input_size = self.loc_emb_size + self.tim_emb_size, hidden_size = self.hidden_size)

        #dense layer
        self.dense = nn.Linear(in_features = self.hidden_size, out_features = self.loc_size)

    def forward(self, batch):
        #input
        # batch*seq *input_size
        loc = batch['current_loc']
        tim = batch['current_tim']
        user = batch['uid']

        # batch*seq *emb_size
        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        user_emb = self.emb_user(user)

        attrs_latent = torch.cat([loc_emb, tim_emb], dim = 2).permute(1, 0, 2)#change batch*seq*emb_size to seq*batch*emb_size
        lstm_out = self.lstm(attrs_latent)#seq*batch*hidden_size

        dense = self.dense(lstm_out).permute(1, 0, 2)#change seq*batch*loc_size to batch*seq*loc_size

        out_vec = torch.add(dense, user_emb)#batch*seq*loc_size

        pred = nn.Softmax(dim = 2)(out_vec)#result batch*seq*loc_size
        return pred

    def predict(self, batch):
        return self.forawrd(batch)

    def calculate_loss(self, batch):                                     
        criterion = nn.NLLLoss()# CrossEntropyLoss: 1)log softmax 2)cross entropy
        scores = self.forward(batch)
        return criterion(scores, batch['target'])


	













