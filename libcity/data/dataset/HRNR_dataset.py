import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.cluster import SpectralClustering
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset

from libcity.data.dataset import TrafficStateDataset
from libcity.model.road_representation.HRNR import SPGAT as GAT
from libcity.model.road_representation.HRNR import GraphConvolution as GCN
from libcity.utils import ensure_dir


class HRNRDataset(TrafficStateDataset):
    def __init__(self, config):
        super().__init__(config)
        torch.autograd.set_detect_anomaly(True)

        self.config = config
        self.dataset = self.config.get("dataset", "")
        self.cache_dataset = self.config.get("cache_dataset", True)  # TODO: save cached dataset

        self.data_path = './raw_data/' + self.dataset + '/'
        if not os.path.exists(self.data_path):
            raise ValueError("Dataset {} not exist! Please ensure the path "
                             "'./raw_data/{}/' exist!".format(self.dataset, self.dataset))

        self.cache_file_folder = "./libcity/cache/dataset_cache"
        ensure_dir(self.cache_file_folder)
        self.geo_file = self.data_path + self.config.get("geo_file", self.dataset) + ".geo"
        self.rel_file = self.data_path + self.config.get("rel_file", self.dataset) + ".rel"

        # HRNR data
        self.node_features = os.path.join(self.cache_file_folder, self.config.get("node_features"))
        self.label_train_set = os.path.join(self.cache_file_folder, self.config.get("label_train_set"))
        self.adj = os.path.join(self.cache_file_folder, self.config.get("adj"))

        self.device = self.config.get('device', torch.device('cpu'))

        self.num_nodes = 0
        self.node_feature_list = []
        self.adj_matrix = None

        self._transfer_files()
        self._calc_transfer_matrix()

    def _transfer_files(self):
        """
        加载.geo .rel，生成HRNR所需的部分文件
        .geo [geo_id, type, coordinates, lane, type, length, bridge]
        .rel [rel_id, type, origin_id, destination_id]
        """
        self._logger.info("generating files...")
        geo = pd.read_csv(self.geo_file)
        rel = pd.read_csv(self.rel_file)
        self.num_nodes = geo.shape[0]
        geo_ids = list(geo["geo_id"])
        self._logger.info("Geo_N is " + str(self.num_nodes))
        feature_dict = {}
        for geo_id in geo_ids:
            feature_dict[geo_id] = [geo_id, "Point", None, geo["lane"][geo_id],
                                    geo["roadtype"][geo_id], geo["length"][geo_id],
                                    geo["bridge"][geo_id]]

        # node_features [[lane, type, length, id]]
        for geo_id in geo_ids:
            node_features = feature_dict[geo_id]
            self.node_feature_list.append(node_features[3:6] + [geo_id])
        pickle.dump(self.node_feature_list, open(self.node_features, "wb"))
        self.node_feature_list = np.array(self.node_feature_list)

        # label_pred_train_set [id]
        is_bridge_ids = []
        for geo_id in geo_ids:
            if str(feature_dict[geo_id][6]) == 1:
                is_bridge_ids.append(geo_id)

        # CompleteAllGraph [[0,1,...,0]]
        self.adj_matrix = [[0 for i in range(0, self.num_nodes)] for j in range(0, self.num_nodes)]
        for row in rel.itertuples():
            origin = getattr(row, "origin_id")
            destination = getattr(row, "destination_id")
            self.adj_matrix[origin][destination] = 1
        pickle.dump(self.adj_matrix, open(self.adj, "wb"))

    def _calc_transfer_matrix(self):
        # calculate T^SR T^RZ with 2 loss functions
        self._logger.info("calculating transfer matrix...")
        self.lane_feature = torch.tensor(self.node_feature_list[:, 0], dtype=torch.long, device=self.device)
        self.type_feature = torch.tensor(self.node_feature_list[:, 1], dtype=torch.long, device=self.device)
        self.length_feature = torch.tensor(self.node_feature_list[:, 2], dtype=torch.long, device=self.device)
        self.node_feature = torch.tensor(self.node_feature_list[:, 3], dtype=torch.long, device=self.device)

        self.node_emb_layer = nn.Embedding(self.config.get("node_num"), self.config.get("node_dims")).to(self.device)
        self.type_emb_layer = nn.Embedding(self.config.get("type_num"), self.config.get("type_dims")).to(self.device)
        self.length_emb_layer = nn.Embedding(self.config.get("length_num"), self.config.get("length_dims")).to(
            self.device)
        self.lane_emb_layer = nn.Embedding(self.config.get("lane_num"), self.config.get("lane_dims")).to(self.device)

        node_emb = self.node_emb_layer(self.node_feature)
        type_emb = self.type_emb_layer(self.type_feature)
        length_emb = self.length_emb_layer(self.length_feature)
        lane_emb = self.lane_emb_layer(self.lane_feature)

        # Segment, Region, Zone dimensions
        self.k1 = self.num_nodes
        self.k2 = self.config.get("struct_cmt_num")
        self.k3 = self.config.get("fnc_cmt_num")
        self._logger.info("k1: " + str(self.k1) + ", k2: " + str(self.k2) + ", k3: " + str(self.k3))

        NS = torch.cat([lane_emb, type_emb, length_emb, node_emb], 1)
        AS = torch.tensor(self.adj_matrix + np.array(np.eye(self.num_nodes)), dtype=torch.float)

        # 谱聚类 求出M1
        self._logger.info("calculating TSR...")
        # sc = SpectralClustering(k2, affinity="precomputed",
        #                         n_init=1, assign_labels="discretize")
        # sc.fit(self.adj_matrix)
        # labels = sc.labels_
        # M1 = [[0 for i in range(k2)] for j in range(k1)]
        # for i in range(k1):
        #     M1[i][labels[i]] = 1
        # M1 = torch.tensor(M1, dtype=torch.long, device=self.device)
        self.hidden_dims = self.config.get("hidden_dims")
        self.dropout = self.config.get("dropout")
        self.alpha = self.config.get("alpha")

        TSR = self.calc_tsr(NS, AS)
        AR = TSR.t().mm(AS).mm(TSR)
        print(torch.max(AR))
        NR = TSR.t().mm(NS)
        TRZ = self.calc_trz(NR, AR, TSR)

        struct_assign = self.config.get("struct_assign", "struct_assign")
        fnc_assign = self.config.get("fnc_assign", "fnc_assign")
        self.struct_assign = torch.tensor(TSR, dtype=torch.float, device=self.device)
        self.fnc_assign = torch.tensor(TRZ, dtype=torch.float, device=self.device)
        pickle.dump(TSR, open(os.path.join(self.cache_file_folder, struct_assign), "wb"))
        pickle.dump(TRZ, open(os.path.join(self.cache_file_folder, fnc_assign), "wb"))

    def calc_tsr(self, NS, AS):
        TSR = None
        sparse_AS = sparse.coo_matrix(AS)
        SR_GAT = GAT(in_features=self.hidden_dims, out_features=self.k2,
                     alpha=self.alpha, dropout=self.dropout).to(self.device)
        loss1 = torch.nn.BCELoss()
        optimizer1 = optim.Adam(SR_GAT.parameters(), lr=5e-2)  # TODO: lr
        optimizer1.zero_grad()
        for i in range(2):  # TODO: 迭代次数
            self._logger.info("epoch " + str(i))
            W1 = SR_GAT(NS, sparse_AS)
            # TSR = W1 * M1
            TSR = W1
            TSR = torch.softmax(TSR, dim=0)

            NR = TSR.t().mm(NS)
            _NS = TSR.mm(NR)
            _AS = torch.sigmoid(_NS.mm(_NS.t()))
            loss = loss1(_AS.reshape(self.k1 * self.k1), AS.reshape(self.k1 * self.k1))
            self._logger.info(" loss: " + str(loss))
            loss.backward(retain_graph=True)
            optimizer1.step()
            optimizer1.zero_grad()
        return TSR

    def calc_trz(self, NR, AR, TSR):
        TRZ = None
        print("NR: " + str(NR))
        print("AR: " + str(AR))
        print("TSR: " + str(TSR))
        # sparse_AR = [[0 for i in range(self.k2)] for i in range(self.k2)]
        # for i in range(self.k2):
        #     for j in range(self.k2):
        #         if AR[i][j] > 0.2:
        #             sparse_AR[i][j] = 1
        #             print((i, j), end=" ")
        # sparse_AR = sparse.coo_matrix(sparse_AR)

        RZ_GCN = GCN(in_features=self.hidden_dims, out_features=self.k3,
                     device=self.device).to(self.device)
        self._logger.info("RZ_GCN: " + str((self.k2, self.hidden_dims))
                          + " -> " + str((self.k2, self.k3)))
        self._logger.info("getting reachable matrix...")
        loss2 = torch.nn.MSELoss()
        optimizer2 = optim.Adam(RZ_GCN.parameters(), lr=5e-2)  # TODO: lr
        optimizer2.zero_grad()
        C = torch.tensor(Utils(self.num_nodes, self.adj_matrix).get_reachable_matrix(), dtype=torch.float)
        self._logger.info("calculating TRZ...")
        for i in range(2):  # TODO: 迭代次数
            self._logger.info("epoch " + str(i))
            TRZ = RZ_GCN(NR.unsqueeze(0), AR.unsqueeze(0)).squeeze()
            TRZ = torch.softmax(TRZ, dim=0)

            NZ = TRZ.t().mm(NR)
            _NS = TSR.mm(TRZ).mm(NZ)
            _C = _NS.mm(_NS.t())
            loss = loss2(C.reshape(self.k1 * self.k1), _C.reshape(self.k1 * self.k1))
            self._logger.info(" loss: " + str(loss))
            loss.backward(retain_graph=True)
            optimizer2.step()
            optimizer2.zero_grad()
        return TRZ

    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据（同测试数据）

        Returns:
            batch_data: dict
        """
        # 正样本和负样本 1:1 提取
        label_pred_train = pickle.load(open(self.label_train_set, "rb"))
        label_pred_train_false = []
        true_sample_cnt = len(label_pred_train)
        while len(label_pred_train_false) < true_sample_cnt:
            x = random.randint(0, self.num_nodes - 1)
            if x not in label_pred_train and x not in label_pred_train_false:
                label_pred_train_false.append(x)
        batch_size = self.config.get("batch_size", 100)
        train_rate = self.config.get("train_rate", 0.7)
        train_index = round(true_sample_cnt * train_rate)

        indexes = []
        labels = []
        for i in range(0, true_sample_cnt):
            indexes += [label_pred_train[i], label_pred_train_false[i]]
            labels += [1, 0]
        train_dataset = TensorDataset(torch.tensor(indexes[0:train_index]), torch.tensor(labels[0:train_index]))
        test_dataset = TensorDataset(torch.tensor(indexes[train_index:]), torch.tensor(labels[train_index:]))

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)
        return train_dataloader, test_dataloader, test_dataloader

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"adj_mx": self.adj_matrix, "num_nodes": self.num_nodes,
                "lane_feature": self.lane_feature, "type_feature": self.type_feature,
                "length_feature": self.length_feature, "node_feature": self.node_feature,
                "struct_assign": self.struct_assign, "fnc_assign": self.fnc_assign}

    def _load_geo(self):
        pass

    def _load_rel(self):
        pass


class Utils:
    def __init__(self, n, adj):
        self.n = n
        self.adj = adj
        self.visited = [False for i in range(self.n)]
        self.t = [[0 for i in range(self.n)] for j in range(self.n)]
        self.temp = []

    def get_reachable_matrix(self):
        # TODO: 使用 osm extract 北京路网，然后用真实的北京下一跳数据
        """
        计算 4.3.2 eq.17 的可达矩阵，只使用邻接矩阵

        Returns:
            列表形式的矩阵
        """
        lam = 3  # lambda in eq.17
        for i in range(0, self.n):
            self.temp = []
            self.dfs(i, i, lam)
            for x in self.temp:
                self.visited[x] = False
        return self.t

    def dfs(self, start, cur, step):
        if step == 0 or self.visited[cur]:
            return
        self.visited[cur] = True
        self.t[start][cur] += 1
        self.temp.append(cur)
        for i in range(self.n):
            if self.adj[cur][i] != 0:
                self.dfs(start, i, step - 1)
