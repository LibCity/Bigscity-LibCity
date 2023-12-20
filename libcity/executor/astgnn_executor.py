import time
import numpy as np
import torch
import os
from libcity.executor.traffic_state_executor import TrafficStateExecutor


class ASTGNNExecutor(TrafficStateExecutor):
    def __init__(self, config, model, data_feature):
        TrafficStateExecutor.__init__(self, config, model, data_feature)
        self.fine_tune_epochs = config.get("fine_tune_epochs", 0)
        self.fine_tune_lr = config.get("fine_tune_lr", 0.001)
        self.raw_epochs = self.epochs
        self.epochs = self.epochs + self.fine_tune_epochs

    def _train_epoch(self, train_dataloader, epoch_idx, loss_func=None):
        """
        该Executor支持train和eval过程的loss计算方式不同
        支持fine tune
        """
        fine_tune = False
        if epoch_idx >= self.raw_epochs:
            # rebuild optimizer
            self.learning_rate = self.fine_tune_lr
            self.optimizer = self._build_optimizer()
            fine_tune = True
        self.model.train()
        loss_func = self.model.calculate_train_loss if not fine_tune else self.model.calculate_val_loss
        losses = []
        for batch in train_dataloader:
            self.optimizer.zero_grad()
            batch.to_tensor(self.device)
            loss = loss_func(batch)
            self._logger.debug(loss.item())
            losses.append(loss.item())
            loss.backward()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        return losses
    
    def _valid_epoch(self, eval_dataloader, epoch_idx, loss_func=None):
        with torch.no_grad():
            self.model.eval()
            loss_func = self.model.calculate_val_loss
            losses = []
            for batch in eval_dataloader:
                batch.to_tensor(self.device)
                loss = loss_func(batch)
                self._logger.debug(loss.item())
                losses.append(loss.item())
            mean_loss = np.mean(losses)
            self._writer.add_scalar('eval loss', mean_loss, epoch_idx)
            return mean_loss
        
    def evaluate(self, test_dataloader):
        self._logger.info('Start evaluating ...')
        with torch.no_grad():
            self.model.eval()
            y_truths = []
            y_preds = []
            for batch in test_dataloader:
                batch.to_tensor(self.device)
                output = self.model.predict(batch)
                labels = self.model.get_label(batch)
                y_true = self._scaler.inverse_transform(labels)
                y_pred = self._scaler.inverse_transform(output)
                y_truths.append(y_true.cpu().numpy())
                y_preds.append(y_pred.cpu().numpy())
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