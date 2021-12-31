from libcity.executor.abstract_executor import AbstractExecutor
from libcity.data.dataset.metapath2vec_dataset import *
from libcity.model.road_representation.Metapath2Vec import *

import os.path as osp
import torch
from numpy import mat
from tensorboardX import SummaryWriter

# 数据集负责将原始rel文件加载为邻接矩阵，测试训练验证数据均由model通过随机游走产生
class Metapath2VecExecutor(AbstractExecutor):
    def __init__(self, config, model):
        self.device = config['device']
        self.model = model.to(self.device)
        # 管道共享张量不在WINDOWS允许
        self.loader = model.loader(batch_size=64, shuffle=True, num_workers=0)
        self.optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
        print('加载数据完成')

    def evaluate(self, test_dataloader): # 测试集
        acc = self._test(test_dataloader)
        print((f'evaluate result: {acc:.4f}'))

    def train(self, train_dataloader, eval_dataloader): # 训练集 和 验证集
        for epoch in range(1, 10):
            self._valid_epoch(epoch,train_dataloader)
            acc = self._test(eval_dataloader)
            print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')
        writer = SummaryWriter('runs/embedding_example')
        writer.add_embedding(mat=self.model('road', batch=train_dataloader['road'].y_index),metadata=train_dataloader['road'].y)

    def save_model(self, cache_name):
        pass

    def load_model(self, cache_name):
        pass
    
    @torch.no_grad()
    def _test(self, test_dataloader,train_ratio=0.1):
        self.model.eval()
        z = self.model(node_type='road', batch=test_dataloader['road'].y_index)  # 调用forward返回embedding
        y = test_dataloader['road'].y # 对应的标签
        perm = torch.randperm(z.size(0))
        train_perm = perm[:int(z.size(0) * train_ratio)]
        test_perm = perm[int(z.size(0) * train_ratio):]

        return self.model.test(z[train_perm], y[train_perm], z[test_perm], y[test_perm],
                        max_iter=150)

    def _valid_epoch(self, epoch, train_dataloader,log_steps=100, eval_steps=2000):
        model = self.model
        loader = self.loader
        optimizer = self.optimizer
        device = self.device

        total_loss = 0
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (i + 1) % log_steps == 0:
                print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                    f'Loss: {total_loss / log_steps:.4f}'))
                total_loss = 0

            if (i + 1) % eval_steps == 0:
                acc = self._test(train_dataloader)
                print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                    f'Acc: {acc:.4f}'))