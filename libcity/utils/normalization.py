import numpy as np
import torch


class Scaler:
    """
    归一化接口
    """

    def transform(self, data):
        """
        数据归一化接口

        Args:
            data(np.ndarray): 归一化前的数据

        Returns:
            np.ndarray: 归一化后的数据
        """
        raise NotImplementedError("Transform not implemented")

    def inverse_transform(self, data):
        """
        数据逆归一化接口

        Args:
            data(np.ndarray): 归一化后的数据

        Returns:
            np.ndarray: 归一化前的数据
        """
        raise NotImplementedError("Inverse_transform not implemented")


class NoneScaler(Scaler):
    """
    不归一化
    """

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class NormalScaler(Scaler):
    """
    除以最大值归一化
    x = x / x.max
    """

    def __init__(self, maxx):
        self.max = maxx

    def transform(self, data):
        return data / self.max

    def inverse_transform(self, data):
        return data * self.max


class StandardScaler(Scaler):
    """
    Z-score归一化
    x = (x - x.mean) / x.std
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class MinMax01Scaler(Scaler):
    """
    MinMax归一化 结果区间[0, 1]
    x = (x - min) / (max - min)
    """

    def __init__(self, minn, maxx):
        self.min = minn
        self.max = maxx

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min


class MinMax11Scaler(Scaler):
    """
    MinMax归一化 结果区间[-1, 1]
    x = (x - min) / (max - min)
    x = x * 2 - 1
    """

    def __init__(self, minn, maxx):
        self.min = minn
        self.max = maxx

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2. - 1.

    def inverse_transform(self, data):
        return ((data + 1.) / 2.) * (self.max - self.min) + self.min


class LogScaler(Scaler):
    """
    Log scaler
    x = log(x+eps)
    """

    def __init__(self, eps=0.999):
        self.eps = eps

    def transform(self, data):
        return np.log(data + self.eps)

    def inverse_transform(self, data):
        return np.exp(data) - self.eps


class StandardIndependCScaler(Scaler):
    """
    Z-score归一化
    每个channel单独进行
    """

    def __init__(self, x_train):
        self.dim = x_train.shape[-1]
        self._channel_mean = []
        self._channel_std = []
        for d in range(self.dim):
            self._channel_mean.append(x_train[..., d].mean())
            self._channel_std.append(x_train[..., d].std())
        self._channel_mean = np.array(self._channel_mean)
        self._channel_std = np.array(self._channel_std)

    def transform(self, data, **kw):
        assert (data.shape[-1] == self.dim), 'Bad channel num for this scalar.'
        return (data - self._channel_mean) / self._channel_std

    def inverse_transform(self, data, **kw):
        if type(data) == torch.Tensor:
            _channel_mean = torch.from_numpy(self._channel_mean).to(data.device)
            _channel_std = torch.from_numpy(self._channel_std).to(data.device)
            _channel_mean.requires_grad = False
            _channel_std.requires_grad = False
        else:
            _channel_mean = self._channel_mean
            _channel_std = self._channel_std
        if kw.__contains__('channel_idx') is False:
            assert (data.shape[-1] == self.dim), 'Bad channel num for this scalar.'
            return (data * _channel_std) + _channel_mean
        elif type(kw['channel_idx']) == list:
            assert (len(kw['channel_idx']) <= self.dim), 'Bad channel num for this scalar.'
            return (data * _channel_std[kw['channel_idx']]) + _channel_mean[kw['channel_idx']]
        else:
            raise TypeError
