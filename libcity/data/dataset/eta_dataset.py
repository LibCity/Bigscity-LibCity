import os
import pandas as pd
import importlib
import json
import math
from tqdm import tqdm

from logging import getLogger

from libcity.data.dataset import AbstractDataset
from libcity.data.utils import generate_dataloader


class ETADataset(AbstractDataset):

    def __init__(self, config):
        self.config = config
        self.cache_file_folder = './libcity/cache/dataset_cache/'
        self.data_path = './raw_data/{}/'.format(self.config['dataset'])
        self.data = None
        self._logger = getLogger()
        # 加载 encoder
        self.encoder = self._get_encoder()
        self.pad_item = None  # 因为若是使用缓存, pad_item 是记录在缓存文件中的而不是 encoder

    def _get_encoder(self):
        try:
            return getattr(importlib.import_module('libcity.data.dataset.eta_encoder'),
                           self.config['eta_encoder'])(self.config)
        except AttributeError:
            raise AttributeError('eta encoder is not found')

    def _load_dyna(self):
        """
        轨迹存储格式: (dict)
            {
                uid: [
                    [
                        dyna_record,
                        dyna_record,
                        ...
                    ],
                    [
                        dyna_record,
                        dyna_record,
                        ...
                    ],
                    ...
                ],
                ...
            }
        """
        # load data according to config
        dyna_file = pd.read_csv(os.path.join(
            self.data_path, '{}.dyna'.format(self.config['dataset'])))
        self._logger.info("Loaded file " + self.config['dataset'] + '.dyna, shape=' + str(dyna_file.shape))
        self.dyna_feature_column = {col: i for i, col in enumerate(dyna_file)}
        res = dict()
        traj_id_set = set()
        for dyna in dyna_file.itertuples():
            traj_id = getattr(dyna, "traj_id")
            if traj_id in traj_id_set:
                continue
            traj_id_set.add(traj_id)

            entity_id = getattr(dyna, "entity_id")
            if entity_id not in res:
                res[entity_id] = []
            rows = dyna_file[dyna_file['traj_id'] == traj_id]

            traj = []
            for _, row in rows.iterrows():
                traj.append(row.tolist())
            res[entity_id].append(traj[:])
        return res

    def _encode_traj(self, data):
        """encode the trajectory

        Args:
            data (dict): the key is uid, the value is the uid's trajectories. For example:
                {
                    uid: [
                        trajectory1,
                        trajectory2
                    ]
                }
                trajectory1 = [
                    checkin_record,
                    checkin_record,
                    .....
                ]

        Return:
            dict: For example:
                {
                    data_feature: {...},
                    pad_item: {...},
                    encoded_data: {uid: encoded_trajectories}
                }
        """
        encoded_data = {}
        for uid in tqdm(data, desc="encoding trajectory"):
            encoded_data[str(uid)] = self.encoder.encode(int(uid), data[uid], self.dyna_feature_column)
        self.encoder.gen_data_feature()
        return {
            "data_feature": self.encoder.data_feature,
            "pad_item": self.encoder.pad_item,
            "encoded_data": encoded_data
        }

    def _divide_data(self):
        """
        return:
            train_data (list)
            eval_data (list)
            test_data (list)
        """
        train_data = []
        eval_data = []
        test_data = []
        train_rate = self.config['train_rate']
        eval_rate = self.config['eval_rate']
        user_set = self.data['encoded_data'].keys()
        for uid in tqdm(user_set, desc="dividing data"):
            encoded_trajectories = self.data['encoded_data'][uid]
            traj_len = len(encoded_trajectories)
            # 根据 traj_len 来划分 train eval test
            train_num = math.ceil(traj_len * train_rate)
            eval_num = math.ceil(
                traj_len * (train_rate + eval_rate))
            train_data += encoded_trajectories[:train_num]
            eval_data += encoded_trajectories[train_num: eval_num]
            test_data += encoded_trajectories[eval_num:]
        return train_data, eval_data, test_data

    def _sort_data(self, data, traj_len_idx, chunk_size):
        chunks = (len(data) + chunk_size - 1) // chunk_size
        # re-arrange indices to minimize the padding
        for i in range(chunks):
            data[i * chunk_size: (i + 1) * chunk_size] = sorted(
                data[i * chunk_size: (i + 1) * chunk_size], key=lambda x: x[traj_len_idx], reverse=True)
        return data

    def get_data(self):
        if self.data is None:
            if self.config['cache_dataset'] and os.path.exists(self.encoder.cache_file_name):
                # load cache
                f = open(self.encoder.cache_file_name, 'r')
                self.data = json.load(f)
                self._logger.info("Loading file " + self.encoder.cache_file_name)
                self.pad_item = self.data['pad_item']
                f.close()
            else:
                self._logger.info("Dataset created")
                dyna_data = self._load_dyna()
                encoded_data = self._encode_traj(dyna_data)
                self.data = encoded_data
                self.pad_item = self.encoder.pad_item
                if self.config['cache_dataset']:
                    if not os.path.exists(self.cache_file_folder):
                        os.makedirs(self.cache_file_folder)
                    with open(self.encoder.cache_file_name, 'w') as f:
                        json.dump(encoded_data, f)
                    self._logger.info('Saved at ' + self.encoder.cache_file_name)
        # TODO: 可以按照uid来划分，也可以全部打乱划分
        train_data, eval_data, test_data = self._divide_data()
        scalar_data_feature = self.encoder.gen_scalar_data_feature(train_data)
        self.data["data_feature"].update(scalar_data_feature)
        sort_by_traj_len = self.config["sort_by_traj_len"]
        if sort_by_traj_len:
            '''
            Divide the data into chunks with size = batch_size * 100
            sort by the length in one chunk
            '''
            traj_len_idx = self.data["data_feature"]["traj_len_idx"]
            chunk_size = self.config['batch_size'] * 100

            train_data = self._sort_data(train_data, traj_len_idx, chunk_size)
            eval_data = self._sort_data(eval_data, traj_len_idx, chunk_size)
            test_data = self._sort_data(test_data, traj_len_idx, chunk_size)
        self._logger.info("Number of train data: {}".format(len(train_data)))
        self._logger.info("Number of eval  data: {}".format(len(eval_data)))
        self._logger.info("Number of test  data: {}".format(len(test_data)))
        return generate_dataloader(
            train_data, eval_data, test_data,
            self.encoder.feature_dict,
            self.config['batch_size'],
            self.config['num_workers'], self.pad_item,
            shuffle=not sort_by_traj_len,
        )

    def get_data_feature(self):
        return self.data['data_feature']
