import importlib
import numpy as np
from torch.utils.data import DataLoader
import copy

from trafficdl.data.list_dataset import ListDataset
from trafficdl.data.batch import Batch


def get_dataset(config):
    """
    according the config['dataset_class'] to create the dataset
    return:
        dataset (Dataset): the loaded dataset
    """
    try:
        return getattr(importlib.import_module('trafficdl.data.dataset'),
                       config['dataset_class'])(config)
    except AttributeError:
        raise AttributeError('dataset_class is not found')


def generate_dataloader(train_data, eval_data, test_data, feature_name,
                        batch_size, num_workers, pad_item=None,
                        pad_max_len=None, shuffle=True,
                        pad_with_last_sample=False):
    """
    Args:
        train_data: (list of input)
        eval_data: (list of input)
        test_data: (list of input)
        data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        feature_name (dict): 描述上面 input 每个元素对应的特征名, 应保证len(feature_name) = len(input)
        batch_size (int)
        num_workers (int)
        pad_item (dict): 用于将不定长的特征补齐到一样的长度，每个特征名作为 key，若某特征名不在该 dict 内则不进行补齐。
        pad_max_len (dict): 用于截取不定长的特征，对于过长的特征进行剪切
        shuffle (bool)
        pad_with_last_sample (bool): 对于若最后一个 batch 不满足 batch_size
        的情况，是否进行补齐（使用最后一个元素反复填充补齐）。
    """
    if pad_with_last_sample:
        num_padding = (batch_size - (len(train_data) %
                                     batch_size)) % batch_size
        data_padding = np.repeat(train_data[-1:], num_padding, axis=0)
        train_data = np.concatenate([train_data, data_padding], axis=0)
        num_padding = (batch_size - (len(eval_data) % batch_size)) % batch_size
        data_padding = np.repeat(eval_data[-1:], num_padding, axis=0)
        eval_data = np.concatenate([eval_data, data_padding], axis=0)
        num_padding = (batch_size - (len(test_data) % batch_size)) % batch_size
        data_padding = np.repeat(test_data[-1:], num_padding, axis=0)
        test_data = np.concatenate([test_data, data_padding], axis=0)
    train_dataset = ListDataset(train_data)
    eval_dataset = ListDataset(eval_data)
    test_dataset = ListDataset(test_data)

    def collator(indices):
        batch = Batch(feature_name, pad_item, pad_max_len)
        for item in indices:
            batch.append(copy.deepcopy(item))
        batch.padding()
        return batch
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, collate_fn=collator,
                                  shuffle=shuffle)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=shuffle)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=False)
    return train_dataloader, eval_dataloader, test_dataloader
