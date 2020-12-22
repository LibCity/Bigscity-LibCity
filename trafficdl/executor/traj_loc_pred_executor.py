import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from trafficdl.executor.abstract_executor import AbstractExecutor
from trafficdl.utils import get_model, get_evaluator

class TrajLocPredExecutor(AbstractExecutor):

    def __init__(self, config):
        self.model = get_model(config)
        self.evaluator = get_evaluator(config)
        self.config = config
        if self.config['use_cuda']:
            self.model = self.model.cuda()
        self.tmp_path = './trafficdl/tmp/checkpoint/'
        self.cache_dir = './trafficdl/cache/model_cache'
        self.evaluate_res_dir = './trafficdl/cache/evaluate_cache'
    
    def train(self, train_dataloader, eval_dataloader):
        if self.config['use_cuda']:
            criterion = nn.NLLLoss().cuda()
        else:
            criterion = nn.NLLLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config['lr'],
                            weight_decay=self.config['L2'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=self.config['lr_step'],
                                                 factor=self.config['lr_decay'], threshold= self.config['schedule_threshold'])
        
        if not os.path.exists(self.tmp_path):
            os.makedirs(self.tmp_path)
        metrics = {}
        metrics['train_loss'] = []
        metrics['accuracy'] = []
        train_total_batch = len(train_dataloader.dataset) / train_dataloader.batch_size
        eval_total_batch = len(eval_dataloader.dataset) / eval_dataloader.batch_size
        lr = self.config['lr']
        for epoch in range(self.config['max_epoch']):
            self.model, avg_loss = self.run(train_dataloader, self.model, self.config['use_cuda'], optimizer, criterion, 
                                        self.config['lr'], self.config['clip'], train_total_batch, self.config['verbose'])
            print('==>Train Epoch:{:0>2d} Loss:{:.4f} lr:{}'.format(epoch, avg_loss, lr))
            metrics['train_loss'].append(avg_loss)
            # eval stage
            avg_loss, avg_acc = self._valid_epoch(eval_dataloader, self.model, self.config['use_cuda'], eval_total_batch, self.config['verbose'], criterion)
            print('==>Test Acc:{:.4f} Loss:{:.4f}'.format(avg_acc, avg_loss))
            metrics['accuracy'].append(avg_acc)
            save_name_tmp = 'ep_' + str(epoch) + '.m'
            torch.save(self.model.state_dict(), self.tmp_path + save_name_tmp)
            scheduler.step(avg_acc)
            lr_last = lr
            lr = optimizer.param_groups[0]['lr']
            if lr_last > lr:
                load_epoch = np.argmax(metrics['accuracy'])
                load_name_tmp = 'ep_' + str(load_epoch) + '.m'
                self.model.load_state_dict(torch.load(self.tmp_path + load_name_tmp))
                print('load epoch={} model state'.format(load_epoch))
            if lr <= 0.9 * 1e-5:
                break
        best = np.argmax(metrics['accuracy'])  # 这个不是最好的一次吗？
        avg_acc = metrics['accuracy'][best]
        load_name_tmp = 'ep_' + str(best) + '.m'
        self.model.load_state_dict(torch.load(self.tmp_path + load_name_tmp))
        # 删除之前创建的临时文件夹
        for rt, dirs, files in os.walk(self.tmp_path):
            for name in files:
                remove_path = os.path.join(rt, name)
                os.remove(remove_path)
        os.rmdir(self.tmp_path)

    def load_model(self, cache_name):
        self.model.load_state_dict(torch.load(cache_name))
    
    def save_model(self, cache_name):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        torch.save(self.model.state_dict(), cache_name)

    def evaluate(self, test_dataloader):
        self.model.train(False)
        test_total_batch = len(test_dataloader.dataset) / test_dataloader.batch_size
        cnt = 0
        for batch in test_dataloader:
            batch.to_tensor(gpu=self.config['use_cuda'])
            scores = self.model(batch)
            evaluate_input = {}
            for i in range(len(batch['uid'])):
                u = batch['uid'][i]
                s = batch['session_id'][i]
                trace_input = {}
                trace_input['loc_true'] = [batch['target'][i].item()]
                trace_input['loc_pred'] = [scores[i].tolist()]
                if u not in evaluate_input:
                    evaluate_input[u] = {}
                evaluate_input[u][s] = trace_input
            cnt += 1
            if cnt % self.config['verbose'] == 0:
                print('finish batch {}/{}'.format(cnt, test_total_batch))
            self.evaluator.evaluate(evaluate_input)
        self.evaluator.save_result(self.evaluate_res_dir)

    def run(self, data_loader, model, use_cuda, optimizer, criterion, lr, clip, total_batch, verbose):
        model.train(True)
        total_loss = []
        cnt = 0
        loc_size = model.loc_size
        for batch in data_loader:
            # use accumulating gradients
            # one batch, one step
            optimizer.zero_grad()
            batch.to_tensor(gpu=use_cuda)
            scores = model(batch)
            loss = criterion(scores, batch['target'])
            loss.backward()
            total_loss.append(loss.data.cpu().numpy().tolist())
            try:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip)
                for p in model.parameters():
                    if p.requires_grad:
                        p.data.add_(-lr, p.grad.data)
            except:
                pass
            optimizer.step()
            cnt += 1
            if cnt % verbose == 0:
                print('finish batch {}/{}'.format(cnt, total_batch))
        avg_loss = np.mean(total_loss, dtype=np.float64)
        return model, avg_loss

    def _valid_epoch(self, data_loader, model, use_cuda, total_batch, verbose, criterion):
        model.train(False)
        total_loss = []
        total_acc = []
        cnt = 0
        loc_size = model.loc_size
        for batch in data_loader:
            batch.to_tensor(gpu=use_cuda)
            scores = model(batch) # batch_size * target_len * loc_size
            loss = criterion(scores, batch['target'])
            total_loss.append(loss.data.cpu().numpy().tolist())
            acc = self.get_acc(batch['target'], scores)
            total_acc.append(acc)
            cnt += 1
            if cnt % verbose == 0:
                print('finish batch {}/{}'.format(cnt, total_batch))
        avg_loss = np.mean(total_loss, dtype=np.float64)
        avg_acc = np.mean(total_acc, dtype=np.float64)
        return avg_loss, avg_acc
    
    def get_acc(self, target, scores, topk = 1):
        """target and scores are torch cuda Variable"""
        target = target.data.cpu().numpy()
        val, idxx = scores.data.topk(topk, 1)
        predx = idxx.cpu().numpy()
        correct_cnt = 0
        for i, p in enumerate(predx):
            t = target[i]
            if t in p:
                correct_cnt += 1
        acc = correct_cnt / target.shape[0]
        return acc
