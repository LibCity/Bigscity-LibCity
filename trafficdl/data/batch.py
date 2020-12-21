
class Batch(object):

    def __init__(self, pad_item=None, pad_max_len=None):
        self.data = None
        self.pad_len = None # 默认是根据 batch 中每个特征最长的长度来补齐，如果某个特征的长度超过了 pad_max_len 则进行剪切
        self.pad_max_len = pad_max_len if pad_max_len != None else {}
        self.pad_item = pad_item if pad_item != None else {}

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
        if self.data == None:
            self.data = {}
            self.pad_len = {}
            for key in item:
                self.data[key] = []
                self.data[key].append(item[key])
                self.pad_len[key] = len(item[key])
        else:
            for key in item:
                if key not in self.data:
                    raise KeyError('when append a item to the batch, raise this error!')
                self.data[key].append(item[key])
                if self.pad_len[key] < len(item[key]):
                    # 保持 pad_len 是最大的
                    self.pad_len[key] = len(item[key])

    def padding(self):
        '''
        进行补齐与截取操作
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
                    self.data[key][i] += [self.pad_item[key]] * (self.pad_len[key] - len(self.data[key][i]))
                else:
                    # 截取的原则是，抛弃前面的点
                    # 因为是时间序列嘛
                    self.data[key][i] = self.data[key][i][-max_len:]
