import importlib
from torch.utils.data import DataLoader

from trafficdl.data.list_dataset import ListDataset
from trafficdl.data.batch import Batch

def get_dataset(config):
    '''
    according the config['dataset_class'] to create the dataset
    return:
        dataset (Dataset): the loaded dataset
    '''
    try:
        return getattr(importlib.import_module('trafficdl.data.dataset'), config['dataset_class'])(config)
    except AttributeError:
        raise AttributeError('dataset_class is not found')

def generate_dataloader(train_data, eval_data, test_data, feature_name, batch_size, num_workers, pad_item=None, pad_max_len=None, shuffle=True):
    '''
    Args:
        train_data, eval_data, test_data (list of input): data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        feature_name (list): 描述上面 input 每个元素对应的特征名, 应保证 len(feature_name) = len(input)
        batch_size (int)
        num_workers (int)
        pad_item (dict): 用于将不定长的特征补齐到一样的长度，每个特征名作为 key，若某特征名不在该 dict 内则不进行补齐。
        pad_max_len (dict): 用于截取不定长的特征，对于过长的特征进行剪切
        shuffle (bool)
    '''
    train_dataset = ListDataset(train_data)
    eval_dataset = ListDataset(eval_data)
    test_dataset = ListDataset(test_data)
    def collator(indices):
        batch = Batch(feature_name, pad_item, pad_max_len)
        for item in indices:
            batch.append(item)
        batch.padding()
        return batch
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collator, shuffle=shuffle) 
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collator, shuffle=shuffle) 
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collator, shuffle=shuffle) 
    return train_dataloader, eval_dataloader, test_dataloader
