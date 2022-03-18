import os
import time
import json
import numpy as np
import torch
from libcity.executor.traffic_state_executor import TrafficStateExecutor
from libcity.model import loss
from functools import partial


class ETAExecutor(TrafficStateExecutor):
    def __init__(self, config, model, data_feature):
        super().__init__(config, model, data_feature)
        self.output_pred = config.get("output_pred", True)
        self.output_dim = None
        self._scalar = None

    def _build_train_loss(self):
        """
        根据全局参数`train_loss`选择训练过程的loss函数
        如果该参数为none，则需要使用模型自定义的loss函数
        注意，loss函数应该接收`Batch`对象作为输入，返回对应的loss(torch.tensor)
        """
        if self.train_loss.lower() == 'none':
            self._logger.warning('Received none train loss func and will use the loss func defined in the model.')
            return None
        if self.train_loss.lower() not in ['mae', 'mse', 'rmse', 'mape', 'logcosh', 'huber', 'quantile', 'masked_mae',
                                           'masked_mse', 'masked_rmse', 'masked_mape', 'r2', 'evar']:
            self._logger.warning('Received unrecognized train loss function, set default mae loss func.')
        else:
            self._logger.info('You select `{}` as train loss function.'.format(self.train_loss.lower()))

        def func(batch):
            y_true = batch['time']
            if y_true.dim() == 1:
                y_true = y_true.view(-1, 1)
            y_predicted = self.model.predict(batch)
            if self.train_loss.lower() == 'mae':
                lf = loss.masked_mae_torch
            elif self.train_loss.lower() == 'mse':
                lf = loss.masked_mse_torch
            elif self.train_loss.lower() == 'rmse':
                lf = loss.masked_rmse_torch
            elif self.train_loss.lower() == 'mape':
                lf = loss.masked_mape_torch
            elif self.train_loss.lower() == 'logcosh':
                lf = loss.log_cosh_loss
            elif self.train_loss.lower() == 'huber':
                lf = loss.huber_loss
            elif self.train_loss.lower() == 'quantile':
                lf = loss.quantile_loss
            elif self.train_loss.lower() == 'masked_mae':
                lf = partial(loss.masked_mae_torch, null_val=0)
            elif self.train_loss.lower() == 'masked_mse':
                lf = partial(loss.masked_mse_torch, null_val=0)
            elif self.train_loss.lower() == 'masked_rmse':
                lf = partial(loss.masked_rmse_torch, null_val=0)
            elif self.train_loss.lower() == 'masked_mape':
                lf = partial(loss.masked_mape_torch, null_val=0)
            elif self.train_loss.lower() == 'r2':
                lf = loss.r2_score_torch
            elif self.train_loss.lower() == 'evar':
                lf = loss.explained_variance_score_torch
            else:
                lf = loss.masked_mae_torch
            return lf(y_predicted, y_true)
        return func

    def evaluate(self, test_dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
        with torch.no_grad():
            self.model.eval()
            y_truths = []
            y_preds = []
            test_pred = {}
            for batch in test_dataloader:
                batch.to_tensor(self.device)
                output = self.model.predict(batch)
                y_true = batch['time']
                y_pred = output
                y_truths.append(y_true.cpu().numpy())
                y_preds.append(y_pred.cpu().numpy())
                if self.output_pred:
                    for i in range(y_pred.shape[0]):
                        uid = batch['uid'][i].cpu().long().numpy()[0]
                        if uid not in test_pred:
                            test_pred[str(uid)] = {}
                        traj_id = batch['traj_id'][i].cpu().long().numpy()[0]
                        current_longi = batch['current_longi'][i].cpu().numpy()
                        current_lati = batch['current_lati'][i].cpu().numpy()
                        coordinates = []
                        for longi, lati in zip(current_longi, current_lati):
                            coordinates.append((float(longi), float(lati)))
                        traj_len = batch['traj_len'][i].cpu().long().numpy()[0]
                        start_timestamp = batch['start_timestamp'][i].cpu().long().numpy()[0]
                        outputs = {}
                        outputs['coordinates'] = coordinates[:traj_len]
                        outputs['start_time'] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.localtime(start_timestamp))
                        outputs['truth'] = float(y_true[i].cpu().numpy()[0])
                        outputs['prediction'] = float(y_pred[i].cpu().numpy()[0])
                        test_pred[str(uid)][str(traj_id)] = outputs
            y_preds = np.concatenate(y_preds, axis=0)
            y_truths = np.concatenate(y_truths, axis=0)  # concatenate on batch
            if self.output_pred:
                filename = \
                    time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) + '_' \
                    + self.config['model'] + '_' + self.config['dataset'] + '_predictions.json'
                with open(os.path.join(self.evaluate_res_dir, filename), 'w') as f:
                    json.dump(test_pred, f)
            self.evaluator.clear()
            self.evaluator.collect({'y_true': torch.tensor(y_truths), 'y_pred': torch.tensor(y_preds)})
            test_result = self.evaluator.save_result(self.evaluate_res_dir)
            return test_result
