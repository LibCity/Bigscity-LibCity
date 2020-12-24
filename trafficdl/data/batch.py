import torch
class Batch(object):

    def __init__(self, feature_name, pad_item=None, pad_max_len=None):
        self.data = {}
        self.pad_len = {} # 默认是根据 batch 中每个特征最长的长度来补齐，如果某个特征的长度超过了 pad_max_len 则进行剪切
        self.origin_len = {} # 用于还原补齐
        self.pad_max_len = pad_max_len if pad_max_len != None else {}
        self.pad_item = pad_item if pad_item != None else {}
        self.feature_name = feature_name
        for key in feature_name:
            self.data[key] = []
            if key in self.pad_item:
                self.pad_len[key] = 0
                self.origin_len[key] = []

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        else:
            raise KeyError('{} is not in the batch'.format(key))

    def append(self, item):
        '''
        append a new item into the batch
        Args:
            item (dict): 特征名作为键
        '''
        if len(item) != len(self.feature_name):
            raise KeyError('when append a batch, item is not equal length with feature_name')
        for i in range(len(item)):
            key = self.feature_name[i]
            self.data[key].append(item[i])
            if key in self.pad_item:
                self.origin_len[key].append(len(item[i]))
                if self.pad_len[key] < len(item[i]):
                    # 保持 pad_len 是最大的
                    self.pad_len[key] = len(item[i])

    def padding(self):
        '''
        只提供对一维数组的特征进行补齐
        '''
        for key in self.pad_item:
            # 只对在 pad_item 中的特征进行补齐
            if key not in self.data:
                raise KeyError('when pad a batch, raise this error!')
            max_len = self.pad_len[key]
            if key in self.pad_max_len:
                max_len = min(self.pad_max_len[key], max_len)
            for i in range(len(self.data[key])):
                if len(self.data[key][i]) < max_len:
                    self.data[key][i] += [self.pad_item[key]] * (max_len - len(self.data[key][i]))
                else:
                    # 截取的原则是，抛弃前面的点
                    # 因为是时间序列嘛
                    self.data[key][i] = self.data[key][i][-max_len:]
                    self.origin_len[key][i] = max_len # 对于剪切了的，我们没办法还原，但至少不要使他出错

    def get_origin_len(self, key):
        return self.origin_len[key]

    def to_tensor(self, gpu=True):
        for key in self.data:
            if gpu:
                self.data[key] = torch.FloatTensor(self.data[key]).cuda()
            else:
                self.data[key] = torch.FloatTensor(self.data[key])
