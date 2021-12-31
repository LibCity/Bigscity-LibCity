from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from typing import List, Optional, Tuple
import torch
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor

NodeType = str
EdgeType = Tuple[str, str, str]
OptTensor = Optional[Tensor]
EPS = 1E-15

class Metapath2Vec(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        # num_nodes_dict 每种节点数量
        adj_dict = data_feature['adj_dict'] # 邻接矩阵的稀疏表示
        num_nodes_dict = data_feature['num_nodes_dict'] # 每种节点类型的节点个数

        # 此处需要从config文件中加载
        metapath =  [
            ('road', 'r2r', 'road'),
            ('road', 'r2l', 'link'),
            ('link', 'l2l', 'link'),
            ('link', 'l2r', 'road'),
            ('road', 'r2r', 'road')
        ]
        
        embedding_dim = config.get('embedding_dim') # 维度
        walk_length = config.get('walk_length') # 单次游走长度
        context_size = config.get('context_size') # 上下文窗口
        walks_per_node = config.get('walks_per_node') # 每个结点游走次数
        num_negative_samples = config.get('num_negative_samples') # 负采样个数
        sparse = config.get('sparse') # 是否使用稀疏
        
        assert sparse == True 
        assert walk_length + 1 >= context_size  # 游走长度大于上下文窗口长度
        if walk_length > len(metapath) and metapath[0][0] != metapath[-1][-1]:  # 强制要求metapath起点终点节点类型一致
            raise AttributeError(
                "The 'walk_length' is longer than the given 'metapath', but "
                "the 'metapath' does not denote a cycle")

        self.adj_dict = adj_dict
        self.embedding_dim = embedding_dim
        self.metapath = metapath
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.num_nodes_dict = num_nodes_dict

        # 节点种类
        types = set([x[0] for x in metapath]) | set([x[-1] for x in metapath])
        types = sorted(list(types))
        # 每种节点在embedding中的起止范围
        count = 0
        self.start, self.end = {}, {}
        for key in types:
            self.start[key] = count
            count += num_nodes_dict[key]
            self.end[key] = count
        # 用来给路径中的结点编号加偏移
        offset = [self.start[metapath[0][0]]]
        offset += [self.start[keys[-1]] for keys in metapath
                   ] * int((walk_length / len(metapath)) + 1)  # 每种路径结点偏移
        offset = offset[:walk_length + 1]
        assert len(offset) == walk_length + 1
        self.offset = torch.tensor(offset)

        # + 1 表示用于链接到孤立节点的虚拟节点。
        self.embedding = Embedding(count + 1, embedding_dim, sparse=sparse)
        self.dummy_idx = count  # 节点范围 [0,count]

        self.embedding.reset_parameters()

    def predict(self, node_type: str, batch: OptTensor = None):
        return self.forward(node_type, batch)

    def calculate_loss(self, pos_rw: Tensor, neg_rw: Tensor):
        return self.loss(pos_rw, neg_rw)
        
    def forward(self, node_type: str, batch: OptTensor = None) -> Tensor:
        r"""返回类型为：obj:`node_type`的：obj:`batch`中的节点的嵌入。"""
        emb = self.embedding.weight[self.start[node_type]:self.end[node_type]]
        return emb if batch is None else emb[batch]

    def loader(self, **kwargs):
        r"""返回在异构图上创建正随机游动和负随机游动的数据加载器。

        Args:
            **kwargs (optional): Arguments of
                :class:`torch.utils.data.DataLoader`, such as
                :obj:`batch_size`, :obj:`shuffle`, :obj:`drop_last` or
                :obj:`num_workers`.
        """
        return DataLoader(range(self.num_nodes_dict[self.metapath[0][0]]),  # 只对其中一个结点类型取路径
                          collate_fn=self._sample, **kwargs)

    def _pos_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node)
        # batch = tensor([node_id1,node_id2,node_id1,node_id2,...])
        rws = [batch]  # 起点为batch中的node_id
        # rws = [tensor]
        for i in range(self.walk_length):
            keys = self.metapath[i % len(self.metapath)]  # 取一条metapath，如('author', 'writes', 'paper')
            adj = self.adj_dict[keys]  # 找到邻接矩阵
            batch = self.sample(adj, batch, num_neighbors=1,
                           dummy_idx=self.dummy_idx).view(-1)  # 找到batch中所有结点的下一步
            rws.append(batch)  # 拼接下一步

        rw = torch.stack(rws, dim=-1)  # 最后一维拼接
        rw.add_(self.offset.view(1, -1))  # 广播，加偏移，对应到结点编号
        rw[rw > self.dummy_idx] = self.dummy_idx  # 虚伪节点，但不是很懂他的机制

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size  # 窗口数量，类似卷积核移动
        for j in range(num_walks_per_rw):  # 取窗口
            walks.append(rw[:, j:j + self.context_size])  # 起点 + 起点的上下文窗口
        return torch.cat(walks, dim=0)  # 拼接

    def _neg_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rws = [batch]
        for i in range(self.walk_length):
            keys = self.metapath[i % len(self.metapath)]
            batch = torch.randint(0, self.num_nodes_dict[keys[-1]],
                                  (batch.size(0),), dtype=torch.long)  # 随机在metapath可行节点类型中采样
            rws.append(batch)

        rw = torch.stack(rws, dim=-1)
        rw.add_(self.offset.view(1, -1))

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    # batch 就是 一个batch_size的list，里面是dataset取出来的元素。为节点编号。

    def _sample(self, batch: List[int]) -> Tuple[Tensor, Tensor]:
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch, dtype=torch.long)
        return self._pos_sample(batch), self._neg_sample(batch)

    def loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        r"""使用正负随机游走计算损失"""
        # pos_rw = [rw_num,context_size]
        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()  # start输入节点，rest路径中剩下的节点
        # embedding = [seq_len, batch_size, embedding_size]
        h_start = self.embedding(start).view(pos_rw.size(0), 1,
                                             self.embedding_dim)  # 输入节点向量
        h_rest = self.embedding(rest.view(-1)).view(pos_rw.size(0), -1,
                                                    self.embedding_dim)  # 上下文关联节点向量

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()  # -logσ(Wi * Wo)

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(neg_rw.size(0), 1,
                                             self.embedding_dim)  # 输入节点，同上
        h_rest = self.embedding(rest.view(-1)).view(neg_rw.size(0), -1,
                                                    self.embedding_dim)  # 负采样节点

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()  # -logσ(-Wi * Wu)

        return pos_loss + neg_loss

    def sample(self,src: SparseTensor, subset: Tensor, num_neighbors: int,
               dummy_idx: int) -> Tensor:
        # 掩码矩阵？
        # subset = tensor([1,2,3,4])
        mask = subset < dummy_idx  # 掩码，排除虚节点
        rowcount = torch.zeros_like(subset)  # 下一步个数
        rowcount[mask] = src.storage.rowcount()[subset[mask]]
        mask = mask & (rowcount > 0)
        offset = torch.zeros_like(subset)
        offset[mask] = src.storage.rowptr()[subset[mask]]

        rand = torch.rand((rowcount.size(0), num_neighbors), device=subset.device)
        rand.mul_(rowcount.to(rand.dtype).view(-1, 1))
        rand = rand.to(torch.long)
        rand.add_(offset.view(-1, 1))

        col = src.storage.col()[rand]
        col[~mask] = dummy_idx
        return col

