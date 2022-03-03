import os
import pandas as pd
import math
import copy
from datetime import datetime
import numpy as np
from collections import defaultdict
from torchtext.data import Field
from nltk import ngrams
from tqdm import tqdm
from libcity.data.dataset import AbstractDataset
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from sklearn.neighbors import BallTree


class GeoSANDataset(AbstractDataset):
    def __init__(self, config):
        self.LOD = 17
        dataset_name = config['dataset']
        self.config = config
        raw_dir = "./raw_data"
        dyna = os.path.join(raw_dir, dataset_name, f"{dataset_name}.dyna")
        geo = os.path.join(raw_dir, dataset_name, f"{dataset_name}.geo")
        self.dyna = pd.read_csv(dyna)
        self.geo = pd.read_csv(geo, index_col='geo_id')
        self.loc2gps = {'<pad>': (0.0, 0.0)}
        self.loc2count = {}

        self.n_loc = 1
        self.loc2idx = {'<pad>': 0}
        self.idx2loc = {0: '<pad>'}
        self.idx2gps = {0: (0.0, 0.0)}  # (latitude, longitude) tuple
        self.build_vocab()
        print(f'{self.n_loc} locations')
        self.user_seq, self.user2idx, \
            self.region2idx, self.n_user, \
            self.n_region, self.region2loc, self.n_time = self.processing()
        print(f'{len(self.user_seq)} users')
        print(f'{len(self.region2idx)} regions')

    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据(事实上不提供)、验证数据

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: None(no valid step) \n
                test_dataloader: Dataloader composed of Batch (class)
        """
        assert self.config['executor_config']["train"]["negative_sampler"] == "KNNSampler"
        assert self.config['executor_config']["test"]["negative_sampler"] == "KNNSampler"
        user_visited_locs = self.get_visited_locs()

        train_dataset, test_dataset = self.split()
        batch_size = int(self.config['executor_config']['train']['batch_size'])
        num_workers = int(self.config['executor_config']['train']['num_workers'])
        num_neg = int(self.config['executor_config']['train']['num_negative_samples'])
        print(f"num_neg: {num_neg}")
        print("build LocQuerySystem...")
        loc_query_sys = LocQuerySystem()
        loc_query_sys.build_tree(self)
        # wrap with Dataloader here
        print("get train_loader...")
        sampler = KNNSampler(
            query_sys=loc_query_sys,
            user_visited_locs=user_visited_locs,
            **self.config['executor_config']["train"]["negative_sampler_config"]
        )
        train_loader = DataLoader(train_dataset, sampler=LadderSampler(train_dataset, batch_size),
                                  num_workers=num_workers, batch_size=batch_size,
                                  collate_fn=lambda e: GeoSANDataset.collect_fn_quadkey(e, train_dataset,
                                                                                        sampler, self.QUADKEY,
                                                                                        self.loc2quadkey, k=num_neg))

        test_sampler = KNNSampler(
            query_sys=loc_query_sys,
            user_visited_locs=user_visited_locs,
            **self.config['executor_config']["test"]["negative_sampler_config"]
        )
        print("get test_loader...")
        num_neg_test = int(self.config['executor_config']['test']['num_negative_samples'])
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 collate_fn=lambda e:
                                 GeoSANDataset.collect_fn_quadkey(e, test_dataset,
                                                                  test_sampler, self.QUADKEY,
                                                                  self.loc2quadkey, k=num_neg_test))

        return train_loader, None, test_loader

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        tmp = {
            'nuser': self.n_user,
            'nloc': self.n_loc,
            'ntime': self.n_time,
            'nquadkey': len(self.QUADKEY.vocab.itos)
        }
        return tmp

    @staticmethod
    def collect_fn_quadkey(batch, data_source, sampler, region_processer, loc2quadkey=None, k=5, with_trg_quadkey=True):
        src, trg = zip(*batch)
        user, loc, time, region = [], [], [], []
        data_size = []
        trg_ = []
        trg_probs_ = []
        for e in src:
            u_, l_, t_, r_, b_ = zip(*e)
            data_size.append(len(u_))
            user.append(torch.tensor(u_))
            loc.append(torch.tensor(l_))
            time.append(torch.tensor(t_))
            r_ = region_processer.numericalize(list(r_))  # (L, LEN_QUADKEY)
            region.append(r_)
        user_ = pad_sequence(user, batch_first=True)  # (N,T) 下同，返回时通过.t()变为(T,N)
        loc_ = pad_sequence(loc, batch_first=True)
        time_ = pad_sequence(time, batch_first=True)
        # (T, N, LEN_QUADKEY)
        region_ = pad_sequence(region, batch_first=False)
        if with_trg_quadkey:
            batch_trg_regs = []
            for i, seq in enumerate(trg):
                pos = torch.tensor([[e[1]] for e in seq])
                neg, probs = sampler(seq, k, user=seq[0][0])
                # (L, k+1), k即为负采样的k
                trg_seq = torch.cat([pos, neg], dim=-1)
                trg_.append(trg_seq)
                trg_regs = []
                for trg_seq_idx in range(trg_seq.size(0)):
                    regs = []
                    for loc in trg_seq[trg_seq_idx]:
                        regs.append(loc2quadkey[loc])
                    trg_regs.append(region_processer.numericalize(regs))
                batch_trg_regs.append(torch.stack(trg_regs))
                trg_probs_.append(probs)
            # (N, T, k+1, LEN_QUADKEY)
            batch_trg_regs = pad_sequence(batch_trg_regs, batch_first=True)
            # [(1+k) * T, N, LEN_QUADKEY)
            batch_trg_regs = batch_trg_regs.permute(2, 1,
                                                    0, 3).contiguous().view(-1,
                                                                            batch_trg_regs.size(0),
                                                                            batch_trg_regs.size(3))
            trg_ = pad_sequence(trg_, batch_first=True)
            trg_probs_ = pad_sequence(trg_probs_, batch_first=True, padding_value=1.0)
            trg_ = trg_.permute(2, 1, 0).contiguous().view(-1, trg_.size(0))
            trg_nov_ = [[not e[-1] for e in seq] for seq in trg]
            return user_.t(), loc_.t(), time_.t(), region_, trg_, batch_trg_regs, trg_nov_, trg_probs_, data_size
        else:
            for i, seq in enumerate(trg):
                pos = torch.tensor([[e[1]] for e in seq])
                neg, probs = sampler(seq, k, user=seq[0][0])
                trg_.append(torch.cat([pos, neg], dim=-1))
                trg_probs_.append(probs)
            trg_ = pad_sequence(trg_, batch_first=True)
            trg_probs_ = pad_sequence(trg_probs_, batch_first=True, padding_value=1.0)
            trg_ = trg_.permute(2, 1, 0).contiguous().view(-1, trg_.size(0))
            trg_nov_ = [[not e[-1] for e in seq] for seq in trg]
            return user_.t(), loc_.t(), time_.t(), region_, trg_, trg_nov_, trg_probs_, data_size

    def region_stats(self):
        """
        统计并打印数据集的一些基本信息
        """
        num_reg_locs = []
        for reg in self.region2loc:
            num_reg_locs.append(len(self.region2loc[reg]))
        num_reg_locs = np.array(num_reg_locs, dtype=np.int32)
        print("min #loc/region: {:d}, with {:d} regions".format(np.min(num_reg_locs),
                                                                np.count_nonzero(num_reg_locs == 1)))
        print("max #loc/region:", np.max(num_reg_locs))
        print("avg #loc/region: {:.4f}".format(np.mean(num_reg_locs)))
        hist, bin_edges = np.histogram(num_reg_locs,
                                       bins=[1, 3, 5, 10, 20, 50, 100, 200, np.max(num_reg_locs)])
        for i in range(len(bin_edges) - 1):
            print("#loc in [{}, {}]: {:d} regions".format(math.ceil(bin_edges[i]),
                                                          math.ceil(bin_edges[i + 1] - 1), hist[i]))

    def get_visited_locs(self):
        print("get_visited_locs...")
        user_visited_locs = {}
        for u in range(len(self.user_seq)):
            seq = self.user_seq[u]
            user = seq[0][0]
            user_visited_locs[user] = set()
            for i in reversed(range(len(seq))):
                if not seq[i][4]:
                    break
            user_visited_locs[user].add(seq[i][1])
            seq = seq[:i]
            for check_in in seq:
                user_visited_locs[user].add(check_in[1])
        return user_visited_locs

    def build_vocab(self, min_freq=10):
        for row in tqdm(self.dyna.itertuples(), desc="build_vocab", ncols=100, total=len(self.dyna)):
            loc = getattr(row, 'location')
            coordinate = self.__get_lat_lon__(loc)
            self.add_location(loc, coordinate)

        if min_freq > 0:
            self.n_loc = 1
            self.loc2idx = {'<pad>': 0}
            self.idx2loc = {0: '<pad>'}
            self.idx2gps = {0: (0.0, 0.0)}
            for loc in self.loc2count:
                if self.loc2count[loc] >= min_freq:
                    self.add_location(loc, self.loc2gps[loc])
        self.locidx2freq = np.zeros(self.n_loc - 1, dtype=np.int32)
        for idx, loc in self.idx2loc.items():
            if idx != 0:
                self.locidx2freq[idx - 1] = self.loc2count[loc]

    def processing(self, min_freq=20):
        # 构建user_seq, 每个user对应一个列表，
        # 列表中元素组成：[loc_idx, time_idx, region_idx, region, time]
        user_seq = {}
        region2idx = {}
        idx2region = {}
        regidx2loc = defaultdict(set)
        n_region = 1

        for row in tqdm(self.dyna.itertuples(), desc="processing", ncols=100, total=len(self.dyna)):
            user = getattr(row, 'entity_id')
            loc = getattr(row, 'location')
            lat, lon = self.__get_lat_lon__(loc)
            time = getattr(row, 'time')
            if loc not in self.loc2idx:
                continue
            time = datetime.strptime(time, '%Y-%m-%dT%H:%M:%SZ')
            # time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S+00:00")
            time_idx = time.weekday() * 24 + time.hour + 1
            loc_idx = self.loc2idx[loc]
            region = latlon2quadkey(float(lat), float(lon), self.LOD)
            if region not in region2idx:
                region2idx[region] = n_region
                idx2region[n_region] = region
                n_region += 1
            region_idx = region2idx[region]
            regidx2loc[region_idx].add(loc_idx)
            if user not in user_seq:
                user_seq[user] = list()
            user_seq[user].append([loc_idx, time_idx, region_idx, region, time])

        # 构建user_seq_array, 每个user对应一个列表
        # 列表中元素组成：[user_idx, loc_idx, time_idx, region, is_new_loc]
        # 只有loc数大于min_freq, 且有超过min_freq/2个new_loc时，才会加入到user_seq_array中
        # user2idx：顺序映射原user中加入到user_seq_array的user编号
        user_seq_array = list()
        user2idx = {}
        n_users = 1
        for user, seq in user_seq.items():
            if len(seq) >= min_freq:
                user2idx[user] = n_users
                user_idx = n_users
                seq_new = list()
                tmp_set = set()
                cnt = 0
                for loc, t, _, region_quadkey, _ in sorted(seq, key=lambda e: e[4]):
                    if loc in tmp_set:
                        seq_new.append((user_idx, loc, t, region_quadkey, True))
                    else:
                        seq_new.append((user_idx, loc, t, region_quadkey, False))
                        tmp_set.add(loc)
                        cnt += 1
                if cnt > min_freq / 2:
                    n_users += 1
                    user_seq_array.append(seq_new)

        # 将原region_quadkey替换为按照ngrams=6切分后的quadkey列表
        # 同时添加入all_quadkeys中
        all_quadkeys = []
        for u in range(len(user_seq_array)):
            seq = user_seq_array[u]
            for i in range(len(seq)):
                check_in = seq[i]
                region_quadkey = check_in[3]
                region_quadkey_bigram = ' '.join([''.join(x) for x in ngrams(region_quadkey, 6)])
                region_quadkey_bigram = region_quadkey_bigram.split()
                all_quadkeys.append(region_quadkey_bigram)
                user_seq_array[u][i] = (check_in[0], check_in[1], check_in[2], region_quadkey_bigram, check_in[4])

        # 再把所有的loc对应的quadkey添加到loc2quadkey与all_quadkeys中
        self.loc2quadkey = ['NULL']
        for loc_idx in range(1, self.n_loc):
            lat, lon = self.idx2gps[loc_idx]
            quadkey = latlon2quadkey(float(lat), float(lon), self.LOD)
            quadkey_bigram = ' '.join([''.join(x) for x in ngrams(quadkey, 6)])
            quadkey_bigram = quadkey_bigram.split()
            self.loc2quadkey.append(quadkey_bigram)
            all_quadkeys.append(quadkey_bigram)

        self.QUADKEY = Field(
            sequential=True,
            use_vocab=True,
            batch_first=True,
            unk_token=None,
            preprocessing=str.split
        )
        self.QUADKEY.build_vocab(all_quadkeys)

        return user_seq_array, user2idx, region2idx, n_users, n_region, regidx2loc, 169

    def __get_lat_lon__(self, loc):
        coor = eval(self.geo.loc[loc]['coordinates'])
        return coor[0], coor[1]

    def add_location(self, loc, coordinate):
        if loc not in self.loc2idx:
            self.loc2idx[loc] = self.n_loc
            self.loc2gps[loc] = coordinate
            self.idx2loc[self.n_loc] = loc
            self.idx2gps[self.n_loc] = coordinate
            if loc not in self.loc2count:
                self.loc2count[loc] = 1
            self.n_loc += 1
        else:
            self.loc2count[loc] += 1

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, idx):
        return self.user_seq[idx]

    def split(self, max_len=100):
        print("split dataset...")
        train_ = copy.copy(self)
        test_ = copy.copy(self)
        # 训练数据由(src, trg)组成, src/trg含有的元素(位置信息)均不超过max_len=100
        # 每个元素的格式为[user_idx, loc_idx, time_idx, region, is_new_loc]
        # 其中region为按照ngrams=6切分后的quadkey列表
        train_seq = list()
        test_seq = list()
        for u in range(len(self)):
            seq = self[u]
            i = 0
            # 找到最后一个不曾访问过的loc的索引i
            for i in reversed(range(len(seq))):
                if not seq[i][4]:
                    break
            for b in range(math.floor((i + max_len - 1) // max_len)):
                if (i - b * max_len) > max_len*1.1:
                    trg = seq[(i - (b + 1) * max_len): (i - b * max_len)]
                    src = seq[(i - (b + 1) * max_len - 1): (i - b * max_len - 1)]
                    train_seq.append((src, trg))
                else:
                    trg = seq[1: (i - b * max_len)]
                    src = seq[0: (i - b * max_len - 1)]
                    train_seq.append((src, trg))
                    break
            # test_seq的src的长度为min(i,max_len), 为到i之前的元素；
            # trg的长度为1, 即索引i对应的元素
            test_seq.append((seq[max(0, -max_len+i):i], seq[i:i+1]))
        train_.user_seq = train_seq
        test_.user_seq = sorted(test_seq, key=lambda e: len(e[0]))
        return train_, test_

# utils for dataset


class LocQuerySystem:
    def __init__(self):
        self.coordinates = []
        self.tree = None
        self.knn = None
        self.knn_results = None
        self.radius = None
        self.radius_results = None

    def build_tree(self, dataset):
        """
        构建KNN(基于BallTree实现)，用于sampler中的采样操作
        """
        self.coordinates = np.zeros((len(dataset.idx2gps) - 1, 2), dtype=np.float64)
        for idx, (lat, lon) in dataset.idx2gps.items():
            if idx != 0:
                self.coordinates[idx - 1] = [lat, lon]
        self.tree = BallTree(
            self.coordinates,
            leaf_size=1,
            metric='haversine'
        )

    def prefetch_knn(self, k=100):
        self.knn = k
        self.knn_results = np.zeros((self.coordinates.shape[0], k), dtype=np.int32)
        for idx, gps in tqdm(enumerate(self.coordinates), total=len(self.coordinates), leave=True):
            trg_gps = gps.reshape(1, -1)
            _, knn_locs = self.tree.query(trg_gps, k + 1)
            knn_locs = knn_locs[0, 1:]
            knn_locs += 1
            self.knn_results[idx] = knn_locs

    def prefetch_radius(self, radius=10.0):
        self.radius = radius
        self.radius_results = {}
        radius /= 6371000/1000
        for idx, gps in tqdm(enumerate(self.coordinates), total=len(self.coordinates), leave=True):
            trg_gps = gps.reshape(1, -1)
            nearby_locs = self.tree.query_radius(trg_gps, r=radius)
            nearby_locs = nearby_locs[0]
            nearby_locs = np.delete(nearby_locs, np.where(nearby_locs == idx))
            nearby_locs += 1
            self.radius_results[idx + 1] = nearby_locs

    def get_knn(self, trg_loc, k=100):
        if self.knn is not None and k <= self.knn:
            return self.knn_results[trg_loc - 1][:k]
        trg_gps = self.coordinates[trg_loc - 1].reshape(1, -1)
        _, knn_locs = self.tree.query(trg_gps, k + 1)
        knn_locs = knn_locs[0, 1:]
        knn_locs += 1
        return knn_locs

    def get_radius(self, trg_loc, r=10.0):
        if r == self.radius:
            return self.radius_results[trg_loc]
        r /= 6371000/1000
        trg_gps = self.coordinates[trg_loc - 1].reshape(1, -1)
        nearby_locs = self.tree.query_radius(trg_gps, r=r)
        nearby_locs = nearby_locs[0]
        nearby_locs = np.delete(nearby_locs, np.where(nearby_locs == trg_loc - 1))
        nearby_locs += 1
        return nearby_locs

    def radius_stats(self, radius=10):
        radius /= 6371000/1000
        num_nearby_locs = []
        for gps in tqdm(self.coordinates, total=len(self.coordinates), leave=True):
            trg_gps = gps.reshape(1, -1)
            count = self.tree.query_radius(trg_gps, r=radius, count_only=True)[0]
            num_nearby_locs.append(count)
        num_nearby_locs = np.array(num_nearby_locs, dtype=np.int32)
        max_loc_idx = np.argsort(-num_nearby_locs)[0]
        print("max #nearby_locs: {:d}, at loc {:d}".format(num_nearby_locs[max_loc_idx], max_loc_idx + 1))


class KNNSampler(nn.Module):
    def __init__(self, query_sys, user_visited_locs, num_nearest=100, exclude_visited=False):
        nn.Module.__init__(self)
        self.query_sys = query_sys
        self.num_nearest = num_nearest
        self.user_visited_locs = user_visited_locs
        self.exclude_visited = exclude_visited

    def forward(self, trg_seq, k, user, **kwargs):
        """
            基于query_sys从候选集中随机采样k个作为负样例
        """
        neg_samples = []
        for check_in in trg_seq:
            trg_loc = check_in[1]
            nearby_locs = self.query_sys.get_knn(trg_loc, k=self.num_nearest)
            if not self.exclude_visited:
                samples = np.random.choice(nearby_locs, size=k, replace=True)
            else:
                samples = []
                for _ in range(k):
                    sample = np.random.choice(nearby_locs)
                    while sample in self.user_visited_locs[user]:
                        sample = np.random.choice(nearby_locs)
                    samples.append(sample)
            neg_samples.append(samples)
        neg_samples = torch.tensor(neg_samples, dtype=torch.long)
        probs = torch.ones_like(neg_samples, dtype=torch.float32)
        return neg_samples, probs


EarthRadius = 6378137
MinLatitude = -85.05112878
MaxLatitude = 85.05112878
MinLongitude = -180
MaxLongitude = 180


def clip(n, min_value, max_value):
    return min(max(n, min_value), max_value)


def map_size(level_of_detail):
    return 256 << level_of_detail


def latlon2pxy(latitude, longitude, level_of_detail):
    latitude = clip(latitude, MinLatitude, MaxLatitude)
    longitude = clip(longitude, MinLongitude, MaxLongitude)

    x = (longitude + 180) / 360
    sin_latitude = math.sin(latitude * math.pi / 180)
    y = 0.5 - math.log((1 + sin_latitude) / (1 - sin_latitude)) / (4 * math.pi)

    size = map_size(level_of_detail)
    pixel_x = int(clip(x * size + 0.5, 0, size - 1))
    pixel_y = int(clip(y * size + 0.5, 0, size - 1))
    return pixel_x, pixel_y


def txy2quadkey(tile_x, tile_y, level_of_detail):
    quadkey = []
    for i in range(level_of_detail, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (tile_x & mask) != 0:
            digit += 1
        if (tile_y & mask) != 0:
            digit += 2
        quadkey.append(str(digit))

    return ''.join(quadkey)


def pxy2txy(pixel_x, pixel_y):
    tile_x = pixel_x // 256
    tile_y = pixel_y // 256
    return tile_x, tile_y


def latlon2quadkey(lat, lon, level):
    """
    经纬度 to quadkey 转换函数
    """
    pixel_x, pixel_y = latlon2pxy(lat, lon, level)
    tile_x, tile_y = pxy2txy(pixel_x, pixel_y)
    return txy2quadkey(tile_x, tile_y, level)


class LadderSampler(Sampler):
    def __init__(self, data_source, batch_sz, fix_order=False):
        super(LadderSampler, self).__init__(data_source)
        self.data = [len(e[0]) for e in data_source]
        self.batch_size = batch_sz * 100
        self.fix_order = fix_order

    def __iter__(self):
        if self.fix_order:
            d = zip(self.data, np.arange(len(self.data)), np.arange(len(self.data)))
        else:
            d = zip(self.data, np.random.permutation(len(self.data)), np.arange(len(self.data)))
        d = sorted(d, key=lambda e: (e[1] // self.batch_size, e[0]), reverse=True)
        return iter(e[2] for e in d)

    def __len__(self):
        return len(self.data)
