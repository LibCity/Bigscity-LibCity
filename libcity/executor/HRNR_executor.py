import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data.dataloader import DataLoader

from libcity.executor.traffic_state_executor import TrafficStateExecutor
from libcity.model.road_representation.HRNR import dict_to_object


class HRNRExecutor(TrafficStateExecutor):

    def __init__(self, config, model):
        TrafficStateExecutor.__init__(self, config, model)
        self.loss_func = None

    def train(self, train_dataloader, eval_dataloader: DataLoader):
        self._logger.info("Starting training...")
        hparams = dict_to_object(self.config.config)
        ce_criterion = torch.nn.CrossEntropyLoss()
        max_f1 = 0
        max_auc = 0
        count = 0
        model_optimizer = torch.optim.Adam(self.model.parameters(), lr=hparams.lp_learning_rate)
        eval_dataloader_iter = iter(eval_dataloader)
        for i in range(hparams.label_epoch):
            self._logger.info("epoch " + str(i) + ", processed " + str(count))
            for step, (train_set, train_label) in enumerate(train_dataloader):
                model_optimizer.zero_grad()
                train_set = train_set.clone().detach()
                train_label = train_label.clone().detach()
                pred = self.model(train_set)
                loss = ce_criterion(pred, train_label)
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), hparams.lp_clip)
                model_optimizer.step()
                if count % 20 == 0:
                    eval_data = get_next(eval_dataloader_iter)
                    if eval_data is None:
                        eval_dataloader_iter = iter(eval_dataloader)
                        eval_data = get_next(eval_dataloader_iter)
                    test_set, test_label = eval_data
                    precision, recall, f1, auc = self.test_label_pred(self.model, test_set, test_label, self.device)
                    if auc > max_auc:
                        max_auc = auc
                    if f1 > max_f1:
                        max_f1 = f1
                    self._logger.info("max_auc: " + str(max_auc))
                    self._logger.info("max_f1: " + str(max_f1))
                    self._logger.info("step " + str(count))
                    self._logger.info(loss.item())
                count += 1

    def evaluate(self, test_dataloader):
        for _, (test_set, test_label) in test_dataloader:
            precision, recall, f1, auc = self.test_label_pred(self.model, test_set, test_label, self.device)

    def test_label_pred(self, model, test_set, test_label, device):
        right = 0
        sum_num = 0
        test_set = test_set.clone().detach()
        pred = model(test_set)
        pred_prob = F.softmax(pred, -1)
        pred_scores = pred_prob[:, 1]
        auc = roc_auc_score(np.array(test_label), np.array(pred_scores.tolist()))
        self._logger.info("auc: " + str(auc))

        pred_loc = torch.argmax(pred, 1).tolist()
        right_pos = 0
        right_neg = 0
        wrong_pos = 0
        wrong_neg = 0
        for item1, item2 in zip(pred_loc, test_label):
            if item1 == item2:
                right += 1
                if item2 == 1:
                    right_pos += 1
                else:
                    right_neg += 1
            else:
                if item2 == 1:
                    wrong_pos += 1
                else:
                    wrong_neg += 1
            sum_num += 1
        recall_sum = right_pos + wrong_pos
        precision_sum = wrong_neg + right_pos
        if recall_sum == 0:
            recall_sum += 1
        if precision_sum == 0:
            precision_sum += 1
        recall = float(right_pos) / recall_sum
        precision = float(right_pos) / precision_sum
        if recall == 0 or precision == 0:
            self._logger.info("p/r/f:0/0/0")
            return 0.0, 0.0, 0.0, 0.0
        f1 = 2 * recall * precision / (precision + recall)
        self._logger.info("label prediction @acc @p/r/f: " + str(float(right) / sum_num) + " " + str(precision) +
                          " " + str(recall) + " " + str(f1))
        return precision, recall, f1, auc


def get_next(it):
    res = None
    try:
        res = next(it)
    except StopIteration:
        pass
    return res
