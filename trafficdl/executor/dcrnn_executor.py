import time
import numpy as np
import torch

from trafficdl.executor.traffic_state_executor import TrafficStateExecutor


class DCRNNExecutor(TrafficStateExecutor):
    def __init__(self, config, model):
        TrafficStateExecutor.__init__(self, config, model)

    def train(self, train_dataloader, eval_dataloader):
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = 0
        num_batches = len(train_dataloader)
        self._logger.info("num_batches:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num
        for epoch_idx in range(self._epoch_num, self.epochs):
            start_time = time.time()
            losses, batches_seen = self._train_epoch(train_dataloader, epoch_idx, batches_seen)
            self._writer.add_scalar('training loss', np.mean(losses), batches_seen)
            self._logger.info("epoch complete!")
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self._logger.info("evaluating now!")
            val_loss = self._valid_epoch(eval_dataloader, epoch_idx, batches_seen)
            end_time = time.time()

            if (epoch_idx % self.log_every) == 0:
                if self.lr_scheduler is not None:
                    log_lr = self.lr_scheduler.get_last_lr()[0]
                else:
                    log_lr = self.learning_rate
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, lr: {:.6f}, {:.1f}s'. \
                    format(epoch_idx, self.epochs, batches_seen, np.mean(losses), val_loss,
                           log_lr, (end_time - start_time))
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
        self.load_model_with_epoch(best_epoch)

    def _train_epoch(self, train_dataloader, epoch_idx, batches_seen=None, loss_func=None):
        self.model.train()
        loss_func = loss_func if loss_func is not None else self.model.calculate_loss
        losses = []
        for batch in train_dataloader:
            self.optimizer.zero_grad()
            batch.to_tensor(self.device)
            loss = loss_func(batch, batches_seen)
            self._logger.debug(loss.item())
            losses.append(loss.item())
            batches_seen += 1
            loss.backward()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        return losses, batches_seen

    def _valid_epoch(self, eval_dataloader, epoch_idx, batches_seen=None, loss_func=None):
        with torch.no_grad():
            self.model.eval()
            loss_func = loss_func if loss_func is not None else self.model.calculate_loss
            losses = []
            for batch in eval_dataloader:
                batch.to_tensor(self.device)
                loss = loss_func(batch, batches_seen)
                self._logger.debug(loss.item())
                losses.append(loss.item())
            mean_loss = np.mean(losses)
            self._writer.add_scalar('eval loss', mean_loss, batches_seen)
            return mean_loss
