import numpy as np

from libcity.executor import TrafficStateExecutor
import os
import time
import torch
from ray import tune

from libcity.model import loss


class CosineWarmupScheduler():
    def __init__(self, optimizer, d_model, n_warmup_steps, lr_mul=1.0):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_periodic_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        self._update_lr()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        if n_steps <= self.n_warmup_steps:
            return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))
        else:
            base = (d_model ** -0.5) * n_warmup_steps ** (-0.5) * (1 + np.cos(
                np.pi * ((n_steps - self.n_warmup_steps) % self.n_periodic_steps) / self.n_periodic_steps))
            return base

    def _update_lr(self):
        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class TESTAMExecutor(TrafficStateExecutor):

    def __init__(self, config, model, data_feature):
        self.hidden_size = config.get("hidden_size", 32)
        self.n_warmup_steps = config.get("n_warmup_steps", 4000)
        self.lr_mul = config.get("lr_mul", 1)
        self.is_quantile = config.get("is_quantile", False)
        self.quantile = config.get("quantile", 0.7)
        super().__init__(config, model, data_feature)

    def _build_train_loss(self):
        def default_train_loss(batch):
            """
            默认的模型训练损失函数

            @param batch: 包含数据的 batch 对象
            @return:
            """

            def get_quantile_label(gated_loss, gate, real):
                max_quantile = gated_loss.quantile(self.quantile)
                min_quantile = gated_loss.quantile(1 - self.quantile)
                incorrect = (gated_loss > max_quantile).expand_as(gate)
                correct = ((gated_loss < min_quantile) & (real.permute(0, 2, 3, 1) != 0)).expand_as(gate)
                cur_expert = gate.argmax(dim=-1, keepdim=True)
                not_chosen = gate.topk(dim=-1, k=2, largest=False).indices
                selected = torch.zeros_like(gate).scatter_(-1, cur_expert, 1.0)
                scaling = torch.zeros_like(gate).scatter_(-1, not_chosen, 0.5)
                selected[incorrect] = scaling[incorrect]
                l_worst_avoidance = selected.detach()
                selected = torch.zeros_like(gate).scatter(-1, cur_expert, 1.0) * correct
                l_best_choice = selected.detach()
                return l_worst_avoidance, l_best_choice

            def get_label(ind_loss, gate, real):
                empty_val = (real.permute(0, 2, 3, 1).expand_as(gate)) == 0
                max_error = ind_loss.argmax(dim=-1, keepdim=True)
                cur_expert = gate.argmax(dim=-1, keepdim=True)
                incorrect = max_error == cur_expert
                selected = torch.zeros_like(gate).scatter(-1, cur_expert, 1.0)
                scaling = torch.ones_like(gate)
                scaling[max_error] = 0.
                scaling = scaling / 2 * (1 - selected)
                l_worst_avoidance = torch.where(incorrect, scaling, selected)
                l_worst_avoidance = torch.where(empty_val, torch.zeros_like(gate), l_worst_avoidance)
                l_worst_avoidance = l_worst_avoidance.detach()
                min_error = ind_loss.argmin(dim=-1, keepdim=True)
                correct = min_error == cur_expert
                scaling = torch.ones_like(gate)
                scaling[min_error] = 2.
                scaling = scaling / scaling.sum(dim=-1, keepdim=True)
                l_best_choice = torch.where(correct, selected, scaling)
                l_best_choice = torch.where(empty_val, torch.zeros_like(gate), l_best_choice)
                l_best_choice = l_best_choice.detach()
                return l_worst_avoidance, l_best_choice

            real = batch['y']
            out, gate, res = self.model.predict(batch)
            predict = self._scaler.inverse_transform(out[..., :self.output_dim])
            real = self._scaler.inverse_transform(real)
            # BTNF -> BFNT
            real = real[..., :self.output_dim].transpose(1, 3)
            ind_loss = loss.masked_mae_torch(self._scaler.inverse_transform(res), real.permute(0, 2, 3, 1), 0.0,
                                             reduce=False)
            if self.is_quantile:
                gated_loss = loss.masked_mae_torch(predict, real, reduce=False).permute(0, 2, 3, 1)
                l_worst_avoidance, l_best_choice = get_quantile_label(gated_loss, gate, real)
            else:
                l_worst_avoidance, l_best_choice = get_label(ind_loss, gate, real)
            worst_avoidance = -.5 * l_worst_avoidance * torch.log(gate)
            best_choice = -.5 * l_best_choice * torch.log(gate)
            return ind_loss.mean() + worst_avoidance.mean() + best_choice.mean()

        # select default train loss function if loss function is None
        if self.train_loss.lower() == 'none':
            self._logger.warning('Received none train loss func and will use the default loss func defined in the'
                                 ' TESTAMExecutor.')
            return default_train_loss
        else:
            return super()._build_train_loss()

    def _build_lr_scheduler(self):
        if self.lr_decay:
            if self.lr_scheduler_type.lower() == "cosinewarmupscheduler":
                self._logger.info('You select `{}` lr_scheduler.'.format(self.lr_scheduler_type.lower()))
                return CosineWarmupScheduler(self.optimizer, self.hidden_size, self.n_warmup_steps, self.lr_mul)
            else:
                return super()._build_lr_scheduler()

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

        for epoch_idx in range(self._epoch_num, self.epochs):
            start_time = time.time()
            losses = self._train_epoch(train_dataloader, epoch_idx, self.loss_func)
            t1 = time.time()
            train_time.append(t1 - start_time)
            self._writer.add_scalar('training loss', np.mean(losses), epoch_idx)
            self._logger.info("epoch complete!")

            self._logger.info("evaluating now!")
            t2 = time.time()
            val_loss = self._valid_epoch(eval_dataloader, epoch_idx)
            end_time = time.time()
            eval_time.append(end_time - t2)

            if self.lr_scheduler is not None and self.lr_scheduler_type.lower() != "cosinewarmupscheduler":
                if self.lr_scheduler_type.lower() == 'reducelronplateau':
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            if (epoch_idx % self.log_every) == 0:
                log_lr = self.optimizer.param_groups[0]['lr']
                message = 'Epoch [{}/{}] train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.2f}s'. \
                    format(epoch_idx, self.epochs, np.mean(losses), val_loss, log_lr, (end_time - start_time))
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
        return min_val_loss

    def _train_epoch(self, train_dataloader, epoch_idx, loss_func=None):
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
        i = 0
        for batch in train_dataloader:
            i += 1
            self.optimizer.zero_grad()
            batch.to_tensor(self.device)
            loss = loss_func(batch)
            self._logger.debug(loss.item())
            losses.append(loss.item())
            loss.backward()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if self.lr_decay and self.lr_scheduler_type.lower() == "cosinewarmupscheduler":
                self.lr_scheduler.step_and_update_lr()
            else:
                self.optimizer.step()
        return losses
