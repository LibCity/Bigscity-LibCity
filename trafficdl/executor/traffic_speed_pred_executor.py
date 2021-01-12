import os
import time

import numpy as np
import torch
from logging import getLogger
from torch.utils.tensorboard import SummaryWriter

from trafficdl.executor.abstract_executor import AbstractExecutor
from trafficdl.utils import get_evaluator, ensure_dir

from trafficdl.model import loss


class TrafficSpeedPredExecutor(AbstractExecutor):
    def __init__(self, config, model):
        self.model = model
        self.evaluator = get_evaluator(config)
        self.metrics = config['metrics']
        self.config = config
        if self.config['gpu']:
            self.model = self.model.cuda()

        self.tmp_path = './trafficdl/tmp/checkpoint'
        self.cache_dir = './trafficdl/cache/model_cache'
        self.evaluate_res_dir = './trafficdl/cache/evaluate_cache'
        self.summary_writer_dir = './trafficdl/log/runs'
        ensure_dir(self.tmp_path)
        ensure_dir(self.cache_dir)
        ensure_dir(self.evaluate_res_dir)
        ensure_dir(self.summary_writer_dir)

        self._writer = SummaryWriter(self.summary_writer_dir)
        self._logger = getLogger()
        self.scaler = self.model.get_data_feature().get('scaler')
        self.data_loader = self.model.get_data_feature().get('data_loader')

        self.max_grad_norm = self.config.get('max_grad_norm', 1.)
        self._epoch_num = self.config.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model_with_epoch(self._epoch_num)

    def save_model(self, cache_name):
        ensure_dir(self.cache_dir)
        self._logger.info("Saved model at " + cache_name)
        torch.save(self.model.state_dict(), cache_name)

    def load_model(self, cache_name):
        self._setup_graph()
        self._logger.info("Loaded model at " + cache_name)
        self.model.load_state_dict(torch.load(cache_name))

    def save_model_with_epoch(self, epoch):
        ensure_dir(self.cache_dir)
        config = dict(self.config)
        config['model_state_dict'] = self.model.state_dict()
        config['epoch'] = epoch
        torch.save(config, self.cache_dir+'/epoch%d.tar' % epoch)
        self._logger.info("Saved model at {}".format(epoch))
        return self.cache_dir+'/epoch%d.tar' % epoch

    def load_model_with_epoch(self, epoch):
        self._setup_graph()
        assert os.path.exists(self.cache_dir+'/epoch%d.tar' % epoch), 'Weights at epoch %d not found' % epoch
        checkpoint = torch.load(self.cache_dir+'/epoch%d.tar' % epoch, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(epoch))

    def _setup_graph(self):
        with torch.no_grad():
            self.model = self.model.eval()
            for batch in self.data_loader:
                batch.to_tensor(gpu=self.config['gpu'])
                output = self.model(batch)
                break

    def evaluate(self, test_dataloader):
        self._logger.info('Start evaluating ...')
        with torch.no_grad():
            self.model = self.model.eval()
            self.evaluator.clear()
            y_truths = []
            y_preds = []
            for batch in test_dataloader:
                batch.to_tensor(gpu=self.config['gpu'])
                output = self.model(batch)
                y_true = self.scaler.inverse_transform(batch['y'].cpu().numpy()[..., 0])
                y_pred = self.scaler.inverse_transform(output.cpu().numpy()[..., 0])
                y_truths.append(y_true)
                y_preds.append(y_pred)
                evaluate_input = {'y_true': y_true, 'y_pred': y_pred}
                self.evaluator.collect(evaluate_input)
            self.evaluator.save_result(self.evaluate_res_dir)
            y_preds = np.concatenate(y_preds, axis=0)
            y_truths = np.concatenate(y_truths, axis=0)  # concatenate on batch
            outputs = {'prediction': y_preds, 'truth': y_truths}
            filename = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) \
                       + '_' + self.config['model'] + '_predictions.npz'
            np.savez_compressed(os.path.join(self.evaluate_res_dir, filename), **outputs)
            self.evaluator.clear()
            self.evaluator.collect({'y_true': y_truths, 'y_pred': y_preds})
            self.evaluator.save_result(self.evaluate_res_dir)

    def train(self, train_dataloader, eval_dataloader):
        epochs = self.config.get('epochs', 100)
        min_val_loss = float('inf')
        wait = 0

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.get('base_lr'), eps=self.config.get('epsilon', 1e-8))
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                            milestones=self.config.get('steps'), gamma=self.config.get('lr_decay_ratio', 0.1))
        self._logger.info('Start training ...')

        # this will fail if model is loaded with a changed batch_size
        if len(train_dataloader.dataset) % train_dataloader.batch_size:
            num_batches = len(train_dataloader.dataset) // train_dataloader.batch_size + 1
        else:
            num_batches = len(train_dataloader.dataset) // train_dataloader.batch_size

        self._logger.info("num_batches:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num
        best_epoch = 0
        for epoch_idx in range(self._epoch_num, epochs):
            start_time = time.time()
            losses, batches_seen = self._train_epoch(train_dataloader, epoch_idx, batches_seen, self._compute_loss)
            self._logger.info("epoch complete")
            self.lr_scheduler.step()

            self._logger.info("evaluating now!")
            val_loss = self._valid_epoch(eval_dataloader, epoch_idx, batches_seen, self._compute_loss)
            end_time = time.time()
            self._writer.add_scalar('training loss', np.mean(losses), batches_seen)

            log_every = self.config.get('log_every', 1)
            if (epoch_idx % log_every) == log_every - 1:
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_idx, epochs, batches_seen,
                                           np.mean(losses), val_loss, self.lr_scheduler.get_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message)

            if val_loss < min_val_loss:
                wait = 0
                if self.config.get('save_model', 1):
                    model_file_name = self.save_model_with_epoch(epoch_idx)
                    self._logger.info('Val loss decrease from {:.4f} to {:.4f}, '
                                      'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss
                best_epoch = epoch_idx
            elif val_loss >= min_val_loss:
                wait += 1
                if wait == self.config.get('patience', 50):
                    self._logger.warning('Early stopping at epoch: %d' % epoch_idx)
                    break
        self.model = self.load_model_with_epoch(best_epoch)

    def _train_epoch(self, train_dataloader, epoch_idx, batches_seen, loss_func=None):
        self.model = self.model.train()
        loss_func = loss_func if loss_func is not None else self.model.calculate_loss
        losses = []
        for batch in train_dataloader:
            self.optimizer.zero_grad()
            batch.to_tensor(gpu=self.config['gpu'])
            output = self.model(batch, batches_seen)
            if batches_seen == 0:
                # this is a workaround to accommodate dynamically registered parameters in DCGRUCell
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.get('base_lr'), eps=self.config.get('epsilon', 1e-8))
            loss = loss_func(batch['y'], output)
            self._logger.debug(loss.item())
            losses.append(loss.item())
            batches_seen += 1
            loss.backward()
            # gradient clipping - this does it in place
            if self.config['clip_grad_norm']:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        return losses, batches_seen

    def _valid_epoch(self, eval_dataloader, epoch_idx, batches_seen, loss_func=None):
        with torch.no_grad():
            self.model = self.model.eval()
            loss_func = loss_func if loss_func is not None else self.model.calculate_loss
            losses = []
            for batch in eval_dataloader:
                batch.to_tensor(gpu=self.config['gpu'])
                output = self.model(batch)
                loss = loss_func(batch['y'], output)
                losses.append(loss.item())
            mean_loss = np.mean(losses)
            self._writer.add_scalar('eval loss', mean_loss, batches_seen)
            return mean_loss

    def _compute_loss(self, y_true, y_predicted):
        y_true = self.scaler.inverse_transform(y_true[..., 0])
        y_predicted = self.scaler.inverse_transform(y_predicted[..., 0])
        return loss.masked_mae_torch(y_predicted, y_true, 0)
