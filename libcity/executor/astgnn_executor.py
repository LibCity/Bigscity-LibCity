import time
import numpy as np
import torch
import os
from libcity.model import loss
from functools import partial
from libcity.executor.traffic_state_executor import TrafficStateExecutor


class ASTGNNExecutor(TrafficStateExecutor):
    def __init__(self, config, model, data_feature):
        TrafficStateExecutor.__init__(self, config, model, data_feature)
        self.fine_tune_epochs = config.get("fine_tune_epochs", 1)
    
    def train(self, train_dataloader, eval_dataloader):
        """
        use data to train model with config

        Args:
            train_dataloader(torch.Dataloader): Dataloader
            eval_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = 0
        train_time = []
        eval_time = []
        num_batches = len(train_dataloader)
        self._logger.info("num_batches:{}".format(num_batches))
        for epoch_idx in range(self._epoch_num, self.epochs + self.fine_tune_epochs):
            start_time = time.time()
            if epoch_idx < self.epochs:
                losses = self._train_epoch(train_dataloader, epoch_idx, self.loss_func)
            else:
                losses = self._train_epoch(train_dataloader, epoch_idx, self.loss_func, fine_tune=True)
            t1 = time.time()
            train_time.append(t1 - start_time)
            self._writer.add_scalar('training loss', np.mean(losses), epoch_idx)
            self._logger.info("epoch complete!")

            self._logger.info("evaluating now!")
            t2 = time.time()
            val_loss = self._valid_epoch(eval_dataloader, epoch_idx, self.loss_func)
            end_time = time.time()
            eval_time.append(end_time - t2)

            if self.lr_scheduler is not None:
                if self.lr_scheduler_type.lower() == 'reducelronplateau':
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            if (epoch_idx % self.log_every) == 0:
                log_lr = self.optimizer.param_groups[0]['lr']
                message = 'Epoch [{}/{}] train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.2f}s'.\
                    format(epoch_idx, self.epochs, np.mean(losses), val_loss, log_lr, (end_time - start_time))
                self._logger.info(message)

            if val_loss < min_val_loss:
                wait = 0
                if self.saved:
                    model_file_name = self.save_model_with_epoch(epoch_idx)
                    self._logger.info('Val loss decrease from {:.4f} to {:.4f}, '
                                      'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss
                best_epoch = epoch_idx
            else:
                wait += 1
                if wait == self.patience and self.use_early_stop:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_idx)
                    break
        
        if len(train_time) > 0:
            self._logger.info('Trained totally {} epochs, average train time is {:.3f}s, '
                              'average eval time is {:.3f}s'.
                              format(len(train_time), sum(train_time) / len(train_time),
                                     sum(eval_time) / len(eval_time)))
        if self.load_best_epoch:
            self.load_model_with_epoch(best_epoch)
        return min_val_loss
    
    def _train_epoch(self, train_dataloader, epoch_idx, loss_func=None, fine_tune = False):
        """
        完成模型一个轮次的训练

        Args:
            train_dataloader: 训练数据
            epoch_idx: 轮次数
            loss_func: 损失函数

        Returns:
            list: 每个batch的损失的数组
        """
        self.model.train()
        loss_func = loss_func if loss_func is not None else self.model.calculate_loss
        losses = []
        if fine_tune:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate*0.1)
        for batch in train_dataloader:
            if not fine_tune:
                self.optimizer.zero_grad()
                batch.to_tensor(self.device)
                loss = loss_func(batch)
                self._logger.debug(loss.item())
                losses.append(loss.item())
                loss.backward()
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            else:
                self.optimizer.zero_grad()
                batch.to_tensor(self.device)
                encoder_inputs = batch['En']
                decoder_inputs = batch['De']
                labels = batch['y']
                # encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)
                # decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)
                # labels = labels.unsqueeze(-1)
                predict_length = labels.shape[2]  # T
                
                encoder_output = self.model.encode(encoder_inputs)
                decoder_start_inputs = decoder_inputs[:, :, :1, :]
                decoder_input_list = [decoder_start_inputs]
                
                for step in range(predict_length):
                    decoder_inputs = torch.cat(decoder_input_list, dim=2)
                    predict_output = self.model.decode(decoder_inputs, encoder_output)
                    decoder_input_list = [decoder_start_inputs, predict_output]
                loss_func = torch.nn.L1Loss().to(self.device)
                loss = loss_func(predict_output, labels)
                self._logger.debug(loss.item())
                losses.append(loss.item())
                loss.backward()
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
        return losses
    
    def _valid_epoch(self, eval_dataloader, epoch_idx, loss_func=None):
        """
        完成模型一个轮次的评估

        Args:
            eval_dataloader: 评估数据
            epoch_idx: 轮次数
            loss_func: 损失函数

        Returns:
            float: 评估数据的平均损失值
        """
        with torch.no_grad():
            self.model.eval()
            # loss_func = loss_func if loss_func is not None else self.model.calculate_loss
            loss_func = torch.nn.L1Loss().to(self.device)
            losses = []
            for batch in eval_dataloader:
                batch.to_tensor(self.device)
                encoder_inputs = batch['En'] 
                decoder_inputs = batch['De']
                labels = batch['y']
                predict_length = labels.shape[2]  # T
                encoder_output = self.model.encode(encoder_inputs)
                decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
                decoder_input_list = [decoder_start_inputs]
                for step in range(predict_length):
                    decoder_inputs = torch.cat(decoder_input_list, dim=2)
                    predict_output = self.model.decode(decoder_inputs, encoder_output)
                    decoder_input_list = [decoder_start_inputs, predict_output]
                loss = loss_func(predict_output, labels)
                self._logger.debug(loss.item())
                losses.append(loss.item())
            mean_loss = np.mean(losses)
            self._writer.add_scalar('eval loss', mean_loss, epoch_idx)
            return mean_loss
        
    def evaluate(self, test_dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        def re_max_min_normalization(x, _max, _min):
            x = (x + 1.) / 2.
            x = 1. * x * (_max - _min) + _min
            return x
        self._logger.info('Start evaluating ...')
        with torch.no_grad():
            self.model.eval()
            # self.evaluator.clear()
            y_truths = []
            y_preds = []
            for batch in test_dataloader:
                batch.to_tensor(self.device)
                encoder_inputs = batch['En']
                decoder_inputs = batch['De']
                labels = batch['y']
                predict_length = labels.shape[2]  # T
                
                encoder_output = self.model.encode(encoder_inputs)
                decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
                decoder_input_list = [decoder_start_inputs]
                
                # 按着时间步进行预测
                for step in range(predict_length):
                    decoder_inputs = torch.cat(decoder_input_list, dim=2)
                    predict_output = self.model.decode(decoder_inputs, encoder_output)
                    decoder_input_list = [decoder_start_inputs, predict_output]
                
                y_true = re_max_min_normalization(batch['y'][..., :self.output_dim].cpu().numpy(), self._scaler.max[0, 0, 0, 0], self._scaler.min[0, 0, 0, 0])
                y_pred = re_max_min_normalization(predict_output[..., :self.output_dim].cpu().numpy(), self._scaler.max[0, 0, 0, 0], self._scaler.min[0, 0, 0, 0])
                # y_pred = self._scaler.inverse_transform(predict_output[..., :self.output_dim])
                y_truths.append(y_true)
                y_preds.append(y_pred)
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