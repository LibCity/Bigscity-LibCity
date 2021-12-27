from libcity.evaluator import TrafficStateEvaluator
from libcity.model import loss
from libcity.evaluator import eval_funcs


class TrafficAccidentEvaluator(TrafficStateEvaluator):

    def __init__(self, config):
        super(TrafficAccidentEvaluator, self).__init__(config)
        self.topk = self.config.get('topk', 10)

    def _check_config(self):
        if not isinstance(self.metrics, list):
            raise TypeError('Evaluator type is not list')
        self.allowed_metrics = ["MAE", "MAPE", "MSE", "RMSE", "masked_MAE", "masked_MAPE", "masked_MSE", "masked_RMSE", "R2", "EVAR",
                                    "Precision", "Recall", "F1-Score", "MAP", "PCC"]
        for metric in self.metrics:
            if metric not in self.allowed_metrics:
                raise ValueError('the metric {} is not allowed in TrafficAccidentEvaluator'.format(str(metric)))

    def collect(self, batch):
        """
        收集一 batch 的评估输入

        Args:
            batch(dict): 输入数据，字典类型，包含两个Key:(y_true, y_pred):
                batch['y_true']: (num_samples/batch_size, timeslots, ..., feature_dim)
                batch['y_pred']: (num_samples/batch_size, timeslots, ..., feature_dim)
        """
        if not isinstance(batch, dict):
            raise TypeError('evaluator.collect input is not a dict of user')
        y_true = batch['y_true']  # tensor
        y_pred = batch['y_pred']  # tensor
        if y_true.shape != y_pred.shape:
            raise ValueError("batch['y_true'].shape is not equal to batch['y_pred'].shape")
        self.len_timeslots = y_true.shape[1]
        for i in range(1, self.len_timeslots + 1):
            for metric in self.metrics:
                if metric + '@' + str(i) not in self.intermediate_result:
                    self.intermediate_result[metric + '@' + str(i)] = []
        if self.mode.lower() == 'average':  # 前i个时间步的平均loss
            for i in range(1, self.len_timeslots + 1):
                for metric in self.metrics:
                    if metric == 'masked_MAE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mae_torch(y_pred[:, :i], y_true[:, :i], 0).item())
                    elif metric == 'masked_MSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mse_torch(y_pred[:, :i], y_true[:, :i], 0).item())
                    elif metric == 'masked_RMSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_rmse_torch(y_pred[:, :i], y_true[:, :i], 0).item())
                    elif metric == 'masked_MAPE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mape_torch(y_pred[:, :i], y_true[:, :i], 0).item())
                    elif metric == 'MAE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mae_torch(y_pred[:, :i], y_true[:, :i]).item())
                    elif metric == 'MSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mse_torch(y_pred[:, :i], y_true[:, :i]).item())
                    elif metric == 'RMSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_rmse_torch(y_pred[:, :i], y_true[:, :i]).item())
                    elif metric == 'MAPE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mape_torch(y_pred[:, :i], y_true[:, :i]).item())
                    elif metric == 'R2':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.r2_score_torch(y_pred[:, :i], y_true[:, :i]).item())
                    elif metric == 'EVAR':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.explained_variance_score_torch(y_pred[:, :i], y_true[:, :i]).item())
        elif self.mode.lower() == 'single':  # 第i个时间步的loss
            for i in range(1, self.len_timeslots + 1):
                for metric in self.metrics:
                    if metric == 'masked_MAE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mae_torch(y_pred[:, i - 1], y_true[:, i - 1], 0).item())
                    elif metric == 'masked_MSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mse_torch(y_pred[:, i - 1], y_true[:, i - 1], 0).item())
                    elif metric == 'masked_RMSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_rmse_torch(y_pred[:, i - 1], y_true[:, i - 1], 0).item())
                    elif metric == 'masked_MAPE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mape_torch(y_pred[:, i - 1], y_true[:, i - 1], 0).item())
                    elif metric == 'MAE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mae_torch(y_pred[:, i - 1], y_true[:, i - 1]).item())
                    elif metric == 'MSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mse_torch(y_pred[:, i - 1], y_true[:, i - 1]).item())
                    elif metric == 'RMSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_rmse_torch(y_pred[:, i - 1], y_true[:, i - 1]).item())
                    elif metric == 'MAPE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mape_torch(y_pred[:, i - 1], y_true[:, i - 1]).item())
                    elif metric == 'R2':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.r2_score_torch(y_pred[:, i - 1], y_true[:, i - 1]).item())
                    elif metric == 'EVAR':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.explained_variance_score_torch(y_pred[:, i - 1], y_true[:, i - 1]).item())
                    elif metric == 'Precision':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            eval_funcs.Precision_torch(y_pred[:, i - 1], y_true[:, i - 1], self.topk))
                    elif metric == 'Recall':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            eval_funcs.Recall_torch(y_pred[:, i - 1], y_true[:, i - 1], self.topk))
                    elif metric == 'F1-Score':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            eval_funcs.F1_Score_torch(y_pred[:, i - 1], y_true[:, i - 1], self.topk))
                    elif metric == 'MAP':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            eval_funcs.MAP_torch(y_pred[:, i - 1], y_true[:, i - 1], self.topk))
                    elif metric == 'PCC':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            eval_funcs.PCC_torch(y_pred[:, i - 1], y_true[:, i - 1], self.topk))
        else:
            raise ValueError('Error parameter evaluator_mode={}, please set `single` or `average`.'.format(self.mode))
