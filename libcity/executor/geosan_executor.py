import torch
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
import time as Time
from libcity.executor.abstract_executor import AbstractExecutor
import random
from libcity.utils import get_evaluator


class GeoSANExecutor(AbstractExecutor):

    def __init__(self, config, model, data_feature):
        self.config = config
        self.device = self.config.get('device', torch.device('cpu'))
        self.model = model.to(self.device)
        self.evaluator = get_evaluator(config)
        self.exp_id = self.config.get('exp_id', None)
        self.cache_dir = './libcity/cache/{}/model_cache'.format(self.exp_id)
        self.evaluate_res_dir = './libcity/cache/{}/evaluate_cache'.format(self.exp_id)
        self.tmp_path = './libcity/tmp/checkpoint/'

    def train(self, train_dataloader, eval_dataloader):
        """
        use data to train model with config

        Args:
            train_dataloader(torch.Dataloader): Dataloader
            eval_dataloader(torch.Dataloader): None
        """
        if not os.path.exists(self.tmp_path):
            os.makedirs(self.tmp_path)
        num_epochs = self.config['executor_config']['train']['num_epochs']
        optimizer = optim.Adam(self.model.parameters(),
                               lr=float(self.config['executor_config']['optimizer']['learning_rate']),
                               betas=(0.9, 0.98))
        self.model.train()
        for epoch_idx in range(num_epochs):
            start_time = Time.time()
            running_loss = 0.
            processed_batch = 0
            batch_iterator = tqdm(enumerate(train_dataloader),
                                  total=len(train_dataloader), leave=True)
            for batch_idx, batch in batch_iterator:
                optimizer.zero_grad()
                loss = self.model.calculate_loss(batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                processed_batch += 1
                batch_iterator.set_postfix_str(f"loss={loss.item():.4f}")
            save_name_tmp = 'ep_' + str(epoch_idx) + '.m'
            torch.save(self.model.state_dict(), self.tmp_path + save_name_tmp)
            epoch_time = Time.time() - start_time
            print("epoch {:>2d} completed.".format(epoch_idx + 1))
            print("time taken: {:.2f} sec".format(epoch_time))
            print("avg. loss: {:.4f}".format(running_loss / processed_batch))
            print("epoch={:d}, loss={:.4f}".format(epoch_idx + 1, running_loss / processed_batch))
        for rt, dirs, files in os.walk(self.tmp_path):
            for name in files:
                remove_path = os.path.join(rt, name)
                os.remove(remove_path)
        os.rmdir(self.tmp_path)
        print("training completed!")

    def evaluate(self, test_dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self.evaluator.clear()
        self.model.eval()
        GeoSANExecutor.reset_random_seed(42)
        with torch.no_grad():
            for _, batch in enumerate(test_dataloader):
                output = self.model.predict(batch)
                # shape: [(1+K)*L, N]
                self.evaluator.collect(output)
        self.evaluator.save_result(self.evaluate_res_dir)

    def load_model(self, cache_name):
        """
        加载对应模型的 cache

        Args:
            cache_name(str): 保存的文件名
        """
        self.model.load(cache_name)

    def save_model(self, cache_name):
        """
        将当前的模型保存到文件

        Args:
            cache_name(str): 保存的文件名
        """
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.model.save(cache_name)

    @staticmethod
    def reset_random_seed(seed):
        """
        重置随机数种子

        Args:
            seed(int): 种子数
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
