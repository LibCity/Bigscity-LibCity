from libcity.data.dataset.abstract_dataset import AbstractDataset
from typing import Dict, Tuple
import pandas as pd
import os.path as osp
import torch
from torch_sparse import coalesce, transpose, SparseTensor


def _typeName(highway):
    highway = str(highway)
    if highway == '3' or highway == '5' or highway == '9' or highway == '11' or highway == '13':
        return 'link'
    return 'road'

def _getType(x):
    type_origin = _typeName(x['origin_highway'])
    type_des = _typeName(x['destination_highway'])
    if type_origin == 'link' and type_des == 'link':
        return 'l2l'
    elif type_origin == 'link' and type_des == 'road':
        return 'l2r'
    elif type_origin == 'road' and type_des == 'link':
        return 'r2l'
    return 'r2r'

class Metapath2VecDataSet(AbstractDataset):
    def __init__(self) -> None:
        self.adj_dict = None
        self.edge_index_dict = None
        self.label_index_dict = None
        self.num_nodes_dict = None
        self.raw_dir = "./raw_data/bj_roadmap_edge/"
        self._process_egdes()
        self._process_features()

    # 返回图的特征
    def get_data_feature(self) -> Dict:
        return {'adj_dict':self.adj_dict,'num_nodes_dict':self.num_nodes_dict}

    def get_data(self) -> Dict: 
        return self.label_index_dict

    def _load_geo(self):
        # Get road labels.
        path = osp.join(self.raw_dir, 'bj_roadmap_edge.geo')
        road = pd.read_csv(path, sep=',')
        road = road[['geo_id','highway']] # 不需要其他特征
        return road

    def _load_rel(self):
        path = osp.join(self.raw_dir, 'bj_roadmap_edge.rel')
        rel = pd.read_csv(path, sep=',', index_col=0)      
        rel = rel[['origin_id','destination_id']] # 不需要type列
        return rel

    def _process_egdes(self):
        edge_index_dict = {} # 保存metapath对应的边
        label_index_dict = {} # 保存标签
        rawgeo = self._load_geo() # geo_id, highway
        rawrel = self._load_rel() # origin_id, destination_id
        # 构建有向图
        rawrel_r = rawrel.copy(deep=True)
        rawrel_r[['origin_id', 'destination_id']] = rawrel_r[['destination_id', 'origin_id']]
        rawrel = pd.concat([rawrel,rawrel_r],axis=0,ignore_index=True)
        # 合并rel表和geo表
        total = pd.merge(rawrel,rawgeo,left_on='origin_id',right_on='geo_id')
        total = total[['origin_id', 'destination_id','highway']]
        total.columns = ['origin_id', 'destination_id','origin_highway']
        total = pd.merge(total,rawgeo,left_on='destination_id',right_on='geo_id')
        total = total[['origin_id', 'destination_id','origin_highway','highway']]
        total.columns = ['origin_id', 'destination_id','origin_highway','destination_highway']
        # 添加新列表示关系类型
        total.loc[:,"rel_type"] = total.apply(_getType,axis=1)
        # total = [ origin_id, destination_id,origin_highway,destination_highway, rel_type]
        # Get road
        data_road = BaseData()
        road = rawgeo.query(
            'highway==1 | highway==2 | highway == 4 | highway == 6 | highway == 7| highway == 8| highway == 10| highway == 12')
        road.columns = ['road_id','road_highway']
        data_road.y = torch.from_numpy(road['road_highway'].values)
        data_road.y_index = torch.from_numpy(road['road_id'].values)
        label_index_dict['road'] = data_road

        # Get link
        data_link = BaseData()
        link = rawgeo.query('highway==3 | highway==5 | highway == 9 | highway == 11 | highway == 13')
        link.columns = ['link_id','link_highway']
        data_link.y = torch.from_numpy(link['link_highway'].values)
        data_link.y_index = torch.from_numpy(link['link_id'].values)
        label_index_dict['link'] = data_link       

        # Get road<->road connectivity.
        road2road = total[total['rel_type'] == 'r2r']
        road2road = torch.from_numpy(road2road[['origin_id','destination_id']].values)
        road2road = road2road.t().contiguous()
        M, N = int(road2road[0].max() + 1), int(road2road[1].max() + 1) # 计数
        road2road, _ = coalesce(road2road, None, M, N)
        label_index_dict['road'].num_nodes = max(N,M)
        edge_index_dict['road', 'r2r', 'road'] = road2road

        # 合并road和link的关系
        link2road = total[total['rel_type'] == 'l2r']
        link2road_r =  link2road.copy(deep=True)
        # 交换两边
        link2road_r[['origin_id','destination_id']] = link2road_r[['destination_id','origin_id']]
        link2road = torch.from_numpy(link2road[['origin_id','destination_id']].values)
        link2road_r = torch.from_numpy(link2road_r[['origin_id','destination_id']].values)
        
        road2link = total[total['rel_type'] == 'r2l']
        road2link_r =  road2link.copy(deep=True)
        road2link_r[['origin_id','destination_id']] = road2link_r[['destination_id','origin_id']]
        road2link = torch.from_numpy(road2link[['origin_id','destination_id']].values)
        road2link_r = torch.from_numpy(road2link_r[['origin_id','destination_id']].values)

        l2r = torch.cat((link2road,road2link_r),0)
        r2l = torch.cat((road2link,link2road_r),0)


        l2r = l2r.t().contiguous()
        M, N = int(l2r[0].max() + 1), int(l2r[1].max() + 1) # 计数
        l2r, _ = coalesce(l2r, None, M, N)

        r2l = r2l.t().contiguous()
        M, N = int(r2l[0].max() + 1), int(r2l[1].max() + 1) # 计数
        r2l, _ = coalesce(r2l, None, M, N)

        edge_index_dict['link', 'l2r', 'road'] = l2r
        edge_index_dict['road', 'r2l', 'link'] = r2l

        # Get link<->link connectivity.
        link2link = total[total['rel_type'] == 'l2l']
        link2link = torch.from_numpy(link2link[['origin_id','destination_id']].values)
        link2link = link2link.t().contiguous()
        M, N = int(link2link[0].max() + 1), int(link2link[1].max() + 1) # 计数
        link2link, _ = coalesce(link2link, None, M, N)
        edge_index_dict['link', 'l2l', 'link'] = link2link
        label_index_dict['link'].num_nodes = max(N,M)

        # 保存图
        self.edge_index_dict = edge_index_dict
        self.label_index_dict = label_index_dict

    def _process_features(self):
        # 用Dict[Tuple[str, str, str], Tensor]的异构边数据处理成所需要的
        edge_index_dict = self.edge_index_dict

        # 节点个数
        if(self.num_nodes_dict == None):
            num_nodes_dict = {}
            for keys, edge_index in edge_index_dict.items():
                key = keys[0]
                N = int(edge_index[0].max() + 1)
                num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

                key = keys[-1]
                N = int(edge_index[1].max() + 1)
                num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))
                
            self.num_nodes_dict = num_nodes_dict

        # 稀疏邻接矩阵,用于随机游走
        if(self.adj_dict == None):
            adj_dict = {}
            for keys, edge_index in edge_index_dict.items():  # edge_index = [2, num_edges]
                sizes = (num_nodes_dict[keys[0]], num_nodes_dict[keys[-1]])
                row, col = edge_index  # 这里的row,col分别代表稀疏矩阵非零下标
                adj = SparseTensor(row=row, col=col, sparse_sizes=sizes)  # 行-列存储索引一一对应，sizes存储矩阵大小
                adj = adj.to('cpu')
                adj_dict[keys] = adj
            self.adj_dict = adj_dict

class BaseData():
    def __init__(self):
        self.num_nodes = None
        self.y = None
        self.y_index = None

