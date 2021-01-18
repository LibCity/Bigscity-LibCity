import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from trafficdl.model.abstract_model import AbstractModel

class FPMC(AbstractModel):
    '''
    FPMC 作为序列推荐任务中的比较经典的模型，也可以适用于下一跳预测任务
    Reference: 
    https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/sequential_recommender/fpmc.py
    https://github.com/khesui/FPMC
    '''
    def __init__(self, config, data_feature):
        super(FPMC, self).__init__(config, data_feature)
        self.embedding_size = config['embedding_size']
        self.device = config['device']
        self.uid_size = data_feature['uid_size']
        self.loc_size = data_feature['loc_size']

        # 可以把 FPMC 的那四个矩阵看成 Embedding
        self.UI_emb = nn.Embedding(self.uid_size, self.embedding_size)
        self.IU_emb = nn.Embedding(self.loc_size, self.embedding_size, padding_idx=data_feature['loc_pad'])
        self.LI_emb = nn.Embedding(self.loc_size, self.embedding_size, padding_idx=data_feature['loc_pad'])
        self.IL_emb = nn.Embedding(self.loc_size, self.embedding_size, padding_idx=data_feature['loc_pad'])

        # 暂不采用矩阵复现思路
        # self.UI = nn.Parameter(torch.randn(size=[self.uid_size, self.n_factors]))
        # self.IU = nn.Parameter(torch.randn(size=[self.loc_size, self.n_factors]))
        # self.LI = nn.Parameter(torch.randn(size=[self.loc_size, self.n_factors]))
        # self.IL = nn.Parameter(torch.randn(size=[self.loc_size, self.n_factors]))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
    
    def forward(self, batch):
        # 矩阵复现法，这样写应该需要单独做一个 FPMCExecutor，感觉不太美观，还是尝试一下赵老师那边的复现思路
        # UI_m_IU = torch.matmul(self.UI, self.IU.T) # uid_size * loc_size
        # # 取出 batch 中每个 uid 对应的行
        # UI_m_IU_user = torch.index_select(UI_m_IU, 0, batch['uid']) # batch_size * loc_size
        
        # IL_m_LI = torch.matmul(self.IL, self.LI.T) # loc_size * loc_size
        # # 取出 batch 中每个 last_current_loc 对应的行
        # last_loc_index = torch.LongTensor(batch.get_origin_len('current_len')) - 1 # Markov chain 仅根据最后一个位置来预测，所以要拿出最后一个位置
        # last_loc = torch.gather(batch['current_loc'], dim=1, index=last_loc_index.unsqueeze(1)) # batch_size * 1
        # IL_m_LI_last_loc = torch.index_select(IL_m_LI, 0, last_loc.squeeze()) # batch_size * loc_size
        # scores = UI_m_IU_user + IL_m_LI_last_loc # batch_size * loc_size

        # Embedding 复现思路
    
        last_loc_index = torch.LongTensor(batch.get_origin_len('current_loc')) - 1 # Markov chain 仅根据最后一个位置来预测，所以要拿出最后一个位置
        last_loc_index = last_loc_index.to(self.device)
        last_loc = torch.gather(batch['current_loc'], dim=1, index=last_loc_index.unsqueeze(1)) # batch_size * 1
        
        user_emb = self.UI_emb(batch['uid']) # batch_size * embedding_size
        last_loc_emb = self.LI_emb(last_loc) # batch_size * 1 * embedding_size

        all_iu_emb = self.IU_emb.weight # loc_size * embedding_size
        mf = torch.matmul(user_emb, all_iu_emb.transpose(0,1))

        all_il_emb = self.IL_emb.weight
        fmc = torch.matmul(last_loc_emb, all_il_emb.transpose(0,1))
        fmc = torch.squeeze(fmc, dim=1)
        score = mf + fmc # batch_size * loc_size
        return score

    def predict(self, batch):
        return self.forward(batch)
    
    def calculate_loss(self, batch):
        # 这个 loss 不太对，之后再改
        criterion = nn.NLLLoss().to(self.device)
        scores = self.forward(batch)
        return criterion(scores, batch['target'])
