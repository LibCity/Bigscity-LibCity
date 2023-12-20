import os
import torch
import numpy as np
import pandas as pd
from libcity.model.controldiffeq import natural_cubic_spline_coeffs
from libcity.data.utils import generate_dataloader

from libcity.data.dataset import TrafficStatePointDataset


class STGNCDEDataset(TrafficStatePointDataset):
    def __init__(self, config):
        super().__init__(config)
        self.missing_test = self.config.get('missing_test', True)

    def get_data(self):
        x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []
        if self.data is None:
            self.data = {}
            if self.cache_dataset and os.path.exists(self.cache_file_name):
                x_train, y_train, x_val, y_val, x_test, y_test = self._load_cache_train_val_test()
            else:
                x_train, y_train, x_val, y_val, x_test, y_test = self._generate_train_val_test()
        # 数据归一化
        self.feature_dim = x_train.shape[-1]
        self.ext_dim = self.feature_dim - self.output_dim
        self.scaler = self._get_scalar(self.scaler_type,
                                       x_train[..., :self.output_dim], y_train[..., :self.output_dim])
        self.ext_scaler = self._get_scalar(self.ext_scaler_type,
                                           x_train[..., self.output_dim:], y_train[..., self.output_dim:])
        x_train[..., :self.output_dim] = self.scaler.transform(x_train[..., :self.output_dim])
        y_train[..., :self.output_dim] = self.scaler.transform(y_train[..., :self.output_dim])
        x_val[..., :self.output_dim] = self.scaler.transform(x_val[..., :self.output_dim])
        y_val[..., :self.output_dim] = self.scaler.transform(y_val[..., :self.output_dim])
        x_test[..., :self.output_dim] = self.scaler.transform(x_test[..., :self.output_dim])
        y_test[..., :self.output_dim] = self.scaler.transform(y_test[..., :self.output_dim])
        # self._logger.info("train_data0:"+str(x_train[0].shape))
        if self.normal_external:
            x_train[..., self.output_dim:] = self.ext_scaler.transform(x_train[..., self.output_dim:])
            y_train[..., self.output_dim:] = self.ext_scaler.transform(y_train[..., self.output_dim:])
            x_val[..., self.output_dim:] = self.ext_scaler.transform(x_val[..., self.output_dim:])
            y_val[..., self.output_dim:] = self.ext_scaler.transform(y_val[..., self.output_dim:])
            x_test[..., self.output_dim:] = self.ext_scaler.transform(x_test[..., self.output_dim:])
            y_test[..., self.output_dim:] = self.ext_scaler.transform(y_test[..., self.output_dim:])
        # 把训练集的X和y聚合在一起成为list，测试集验证集同理
        # x_train/y_train: (num_samples, input_length, ..., feature_dim)
        # train_data(list): train_data[i]是一个元组，由x_train[i]和y_train[i]组成
        if self.missing_test == True:
            generator = torch.Generator().manual_seed(56789)
            xs = np.concatenate([x_train, x_val, x_test])
            for xi in xs:
                removed_points_seq = torch.randperm(xs.shape[1], generator=generator)[:int(xs.shape[1] * self.missing_rate)].sort().values
                removed_points_node = torch.randperm(xs.shape[2], generator=generator)[:int(xs.shape[2] * self.missing_rate)].sort().values

                for seq in removed_points_seq:
                    for node in removed_points_node:
                        xi[seq,node] = float('nan')
            x_train = xs[:x_train.shape[0],...] 
            x_val = xs[x_train.shape[0]:x_train.shape[0]+x_val.shape[0],...]
            x_test = xs[-x_test.shape[0]:,...] 
        times = torch.linspace(0,11,12)
        augmented_X_train = []
        augmented_X_train.append(times.unsqueeze(0).unsqueeze(0).repeat(x_train.shape[0],x_train.shape[2],1).unsqueeze(-1).transpose(1,2))
        augmented_X_train.append(torch.Tensor(x_train[..., :]))
        x_train = torch.cat(augmented_X_train, dim=3)
        augmented_X_val = []
        augmented_X_val.append(times.unsqueeze(0).unsqueeze(0).repeat(x_val.shape[0],x_val.shape[2],1).unsqueeze(-1).transpose(1,2))
        augmented_X_val.append(torch.Tensor(x_val[..., :]))
        x_val = torch.cat(augmented_X_val, dim=3)
        augmented_X_test = []
        augmented_X_test.append(times.unsqueeze(0).unsqueeze(0).repeat(x_test.shape[0],x_test.shape[2],1).unsqueeze(-1).transpose(1,2))
        augmented_X_test.append(torch.Tensor(x_test[..., :]))
        x_test = torch.cat(augmented_X_test, dim=3)

        train_coeffs = natural_cubic_spline_coeffs(times, x_train.transpose(1,2))
        valid_coeffs = natural_cubic_spline_coeffs(times, x_val.transpose(1,2))
        test_coeffs = natural_cubic_spline_coeffs(times, x_test.transpose(1,2))
        a,b,c,d = train_coeffs
        print("bairui shapes:"+str(a.shape)+" "+str(b.shape)+" "+str(c.shape)+" "+str(d.shape))

        train_data = torch.utils.data.TensorDataset(*train_coeffs, torch.tensor(y_train))
        self.train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size,
                                                shuffle=True, drop_last=True)

        if len(x_val) == 0:
            self.eval_dataloader = None
        else:
            eval_data = torch.utils.data.TensorDataset(*valid_coeffs, torch.tensor(y_val))
            self.eval_dataloader = torch.utils.data.DataLoader(eval_data, batch_size=self.batch_size,
                                                    shuffle=False, drop_last=True)
        test_data = torch.utils.data.TensorDataset(*test_coeffs, torch.tensor(y_test))
        self.test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size,
                                                shuffle=False, drop_last=False)
        
        # train_data = list(zip(x_train, y_train))
        # eval_data = list(zip(x_val, y_val))
        # test_data = list(zip(x_test, y_test))
        # self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
        #     generate_dataloader(train_data, eval_data, test_data, self.feature_name,
        #                         self.batch_size, self.num_workers, pad_with_last_sample=self.pad_with_last_sample)
        self.num_batches = len(self.train_dataloader)
        self.feature_dim = x_train.shape[-1]
        self.output_dim = y_train.shape[-1]
        
        
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader


    def get_data_feature(self):
        return {"scaler": self.scaler, "adj_mx": self.adj_mx, "ext_dim": self.ext_dim,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
                "output_dim": self.output_dim, "num_batches": self.num_batches}