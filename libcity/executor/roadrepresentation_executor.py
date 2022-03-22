import os
import time

import pandas as pd
import torch
from ray import tune

from libcity.executor.traffic_state_executor import TrafficStateExecutor
from libcity.model import loss
from libcity.data import get_dataset
from libcity.utils import get_model


class RoadRepresentationExecutor(TrafficStateExecutor):
    def __init__(self, config, model):
        TrafficStateExecutor.__init__(self, config, model)
        self.downstream = config.get("downstream", "self_regression")
        self.loss_func = None
        if self.downstream == "self_regression":
            self.loss_func = self.loss_self_regression
            self.encoder = self.model

            decoder_config = self.config.config.copy()
            # TODO
            dataset = get_dataset(config)
            data_feature = dataset.get_data_feature()

            data_feature["feature_dim"] = decoder_config["feature_dim"] = model.output_dim
            data_feature["output_dim"] = decoder_config["output_dim"] = model.input_dim

            self.decoder = get_model(decoder_config, data_feature)
        elif self.downstream == "label_prediction":
            self.loss_func = self.loss_label_prediction
            self.linear = torch.nn.Linear(self.config.get("output_dim"), self.config.get("label_num"))

    def loss_self_regression(self, batch):
        y_true = batch['node_features'].clone()  # N, feature_dim

        inputs = batch['node_features']
        encoder_state = self.encoder(inputs)  # N, output_dim
        y_predicted = self.decoder(encoder_state)  # N, feature_dim

        y_true = self._scaler.inverse_transform(y_true)
        y_predicted = self._scaler.inverse_transform(y_predicted)

        mask = batch['mask']
        return loss.masked_mse_torch(y_predicted[mask], y_true[mask])

    def loss_label_prediction(self, batch):
        model = self.model
        labels = batch['node_labels']
        node_features = model(batch['node_features'])
        mask = batch['mask']
        predict_labels = self.linear(node_features[mask])
        self._logger.info("labels {}, predict {}".format(labels[mask].shape, predict_labels.shape))
        return torch.nn.CrossEntropyLoss()(predict_labels, labels[mask])

    def evaluate(self, test_dataloader):
        """
        use model to test data
        """
        self.evaluator.evaluate()
        node_features = torch.FloatTensor(test_dataloader['node_features']).to(self.device)
        node_labels = None

        if self.downstream == "self_regression":
            node_labels = node_features.clone()
        elif self.downstream == "label_prediction":
            node_labels = torch.LongTensor(pd.get_dummies(test_dataloader['node_labels']).values)

        test_mask = test_dataloader['mask']

        self._logger.info('Start evaluating ...')
        node_labels = self._scaler.inverse_transform(node_labels)
        self._logger.info('node_labels {}'.format(node_labels.shape))

        if self.downstream == "self_regression":
            with torch.no_grad():
                self.model.eval()
                output = self.encoder.predict({'node_features': node_features})
                output = self.decoder.predict({'node_features': output})
                output = self._scaler.inverse_transform(output)
        elif self.downstream == "label_prediction":
            with torch.no_grad():
                self.model.eval()
                output = self.model.predict({'node_features': node_features})
                output = self.linear(output)
                self._logger.info("output {}".format(output.shape))
                output = self._scaler.inverse_transform(output)

        rmse = loss.masked_rmse_torch(output[test_mask], node_labels[test_mask])
        mae = loss.masked_mae_torch(output[test_mask], node_labels[test_mask])
        mape = loss.masked_mape_torch(output[test_mask], node_labels[test_mask])
        self._logger.info('mae={}, map={}, rmse={}'.format(mae.item(), mape.item(), rmse.item()))
        return mae.item(), mape.item(), rmse.item()

    def train(self, train_dataloader, eval_dataloader):
        """
        use data to train model with config
        """
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = 0
        train_time = []
        eval_time = []

        for epoch_idx in range(self._epoch_num, self.epochs):
            start_time = time.time()
            train_loss = self._train_epoch(train_dataloader, epoch_idx, self.loss_func)
            t1 = time.time()
            train_time.append(t1 - start_time)
            self._writer.add_scalar('training loss', train_loss, epoch_idx)
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
                message = 'Epoch [{}/{}] train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.2f}s' \
                    .format(epoch_idx, self.epochs, train_loss, val_loss, log_lr, (end_time - start_time))
                self._logger.info(message)

            if self.hyper_tune:
                # use ray tune to checkpoint
                with tune.checkpoint_dir(step=epoch_idx) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    self.save_model(path)
                # ray tune use loss to determine which params are best
                tune.report(loss=val_loss)

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
            save_list = os.listdir(self.cache_dir)
            for save_file in save_list:
                if '.tar' in save_file:
                    os.remove(os.path.join(self.cache_dir, save_file))
        return min_val_loss

    def _train_epoch(self, train_dataloader, epoch_idx, loss_func=None):
        """
        完成模型一个轮次的训练

        Returns:
            float: 训练集的损失值
        """
        node_features = torch.FloatTensor(train_dataloader['node_features']).to(self.device)
        node_labels = torch.LongTensor(train_dataloader['node_labels']).to(self.device)
        train_mask = train_dataloader['mask']

        self.model.train()
        self.optimizer.zero_grad()
        loss_func = loss_func if loss_func is not None else self.model.calculate_loss
        loss = loss_func({'node_features': node_features, 'node_labels': node_labels, 'mask': train_mask})
        loss.backward()
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss.item()

    def _valid_epoch(self, eval_dataloader, epoch_idx, loss_func=None):
        """
        完成模型一个轮次的评估

        Args:
            eval_dataloader: 评估数据
            epoch_idx: 轮次数
            loss_func: 损失函数

        Returns:
            float: 验证集的损失值
        """
        node_features = torch.FloatTensor(eval_dataloader['node_features']).to(self.device)
        node_labels = torch.LongTensor(eval_dataloader['node_labels']).to(self.device)
        valid_mask = eval_dataloader['mask']

        with torch.no_grad():
            self.model.eval()
            loss_func = loss_func if loss_func is not None else self.model.calculate_loss
            loss = loss_func({'node_features': node_features, 'node_labels': node_labels, 'mask': valid_mask})
            self._writer.add_scalar('eval loss', loss, epoch_idx)
            return loss.item()
