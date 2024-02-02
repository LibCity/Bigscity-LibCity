import os
import time
import numpy as np
import torch
import math

from libcity.executor.traffic_state_executor import TrafficStateExecutor


class STTSNetExecutor(TrafficStateExecutor):
    def __init__(self, config, model, data_feature):
        self.lrf = config.get("lrf", 0.01)
        super().__init__(config, model, data_feature)

    def _build_optimizer(self):
        """
        根据全局参数`learner`选择optimizer
        """
        self._logger.info('You select `{}` optimizer.'.format(self.learner.lower()))
        pg = [p for p in self.model.parameters() if p.requires_grad]
        if self.learner.lower() == 'adam':
            optimizer = torch.optim.Adam(pg, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            optimizer = super()._build_optimizer()
        return optimizer

    def _build_lr_scheduler(self):
        """
        根据全局参数`lr_scheduler`选择对应的lr_scheduler
        """
        lr_scheduler = None
        # handle lambda
        lf = lambda x: ((1 + math.cos(x * math.pi / self.epochs)) / 2) * (1 - self.lrf) + self.lrf
        if self.lr_decay:
            self._logger.info('You select `{}` lr_scheduler.'.format(self.lr_scheduler_type.lower()))
            if self.lr_scheduler_type.lower() == 'lambdalr':
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lr_lambda=lf)
        else:
            lr_scheduler = super()._build_lr_scheduler()
        return lr_scheduler

    def evaluate(self, test_dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
        with torch.no_grad():
            self.model.eval()
            # self.evaluator.clear()
            y_truths = []
            y_preds = []
            for batch in test_dataloader:
                batch.to_tensor(self.device)
                output = self.model.predict(batch)
                y_true = batch['y']
                y_pred = output
                y_truths.append(y_true.cpu().numpy())
                y_preds.append(y_pred.cpu().numpy())
                # evaluate_input = {'y_true': y_true, 'y_pred': y_pred}
                # self.evaluator.collect(evaluate_input)
            # self.evaluator.save_result(self.evaluate_res_dir)
            y_preds = np.concatenate(y_preds, axis=0)
            y_truths = np.concatenate(y_truths, axis=0)  # concatenate on batch
            outputs = {'prediction': y_preds, 'truth': y_truths}
            filename = \
                time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) + '_' \
                + self.config['model'] + '_' + self.config['dataset'] + '_predictions.npz'
            np.savez_compressed(os.path.join(self.evaluate_res_dir, filename), **outputs)
            self.evaluator.clear()
            self.evaluator.collect({'y_true': torch.tensor(y_truths), 'y_pred': torch.tensor(y_preds)})
            test_result = self.evaluator.save_result(self.evaluate_res_dir)
            return test_result
