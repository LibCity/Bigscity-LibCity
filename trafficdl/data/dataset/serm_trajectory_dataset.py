import os
import json
import pandas as pd
import math
import numpy as np

from trafficdl.data.dataset import AbstractDataset
from trafficdl.utils import parse_time, cal_basetime, cal_timeoff
from trafficdl.data.utils import generate_dataloader

allow_dataset = ['foursquare_tky', 'foursquare_nyk']
WORD_VEC_PATH = './raw_data/word_vec/glove.twitter.27B.50d.txt'
parameters = ['dataset', 'min_session_len', 'min_sessions', 'time_window_size']


class SermTrajectoryDataset(AbstractDataset):

    def __init__(self, config):
        self.config = config
        # 检查数据集是否支持语义信息
        if self.config['dataset'] not in allow_dataset:
            raise TypeError('the {} dataset do not support sematic trajectory. \
                only support {}'.format(self.config['dataset'], allow_dataset))
        parameters_str = ''
        for key in parameters:
            if key in self.config:
                parameters_str += '_' + str(self.config[key])
        self.cache_file_name = os.path.join(
            './trafficdl/cache/dataset_cache/', 'serm_trajectory_{}.json \
            '.format(parameters_str))  # 缓存切好的轨迹
        self.cache_file_folder = './trafficdl/cache/dataset_cache/'
        self.data_path = './raw_data/{}/'.format(self.config['dataset'])
        self.data = None
        self.pad_item = None
        self.pad_max_len = {
            'history_loc': self.config['history_len'],
            'history_tim': self.config['history_len']
        }
        self.feature_name = {
            'history_loc': 'int',
            'history_tim': 'int',
            'current_loc': 'int',
            'current_tim': 'int',
            'target': 'int',
            'target_tim': 'int',
            'uid': 'int',
            'text': 'float'
        }

    def get_data(self):
        """
        轨迹比较特殊，原子文件中存储的并不是轨迹而是一个一个点，因此需要先对轨迹进行切割
        """
        if self.data is None:
            if self.config['cache_dataset'] and os.path.exists(
                    self.cache_file_name):
                # load cache
                f = open(self.cache_file_name, 'r')
                self.data = json.load(f)
                loc_pad = self.data['loc_size'] - 1
                tim_pad = self.data['tim_size'] - 1
                self.pad_item = {
                    'current_loc': loc_pad,
                    'history_loc': loc_pad,
                    'current_tim': tim_pad,
                    'history_tim': tim_pad,
                    'text': np.zeros((self.data['text_size']))
                }
                f.close()
            else:
                transformed_data = self.cutter_filter()
                # pad parameter
                loc_pad = transformed_data['loc_size']
                transformed_data['loc_size'] += 1
                tim_pad = transformed_data['tim_size']
                transformed_data['tim_size'] += 1
                self.pad_item = {
                    'current_loc': loc_pad,
                    'history_loc': loc_pad,
                    'current_tim': tim_pad,
                    'history_tim': tim_pad,
                    'text': np.zeros((transformed_data['text_size']))
                }
                self.data = transformed_data
                if self.config['cache_dataset']:
                    if not os.path.exists(self.cache_file_folder):
                        os.makedirs(self.cache_file_folder)
                    with open(self.cache_file_name, 'w') as f:
                        json.dump(transformed_data, f)
        # 切完轨迹之后，就是做 batch 了
        # 划分训练集、测试集、验证集：有两种方式按 user 来划，以及按轨迹数来划。
        # TODO: 这里可以设一个参数，现在先按照轨迹数来划吧
        train_data, eval_data, test_data = self.gen_input()
        return generate_dataloader(
            train_data, eval_data, test_data, self.feature_name,
            self.config['batch_size'], self.config['num_workers'],
            self.pad_item, self.pad_max_len)

    def get_data_feature(self):
        res = {
            'loc_size': self.data['loc_size'],
            'tim_size': self.data['tim_size'],
            'uid_size': self.data['uid_size'],
            'loc_pad': self.pad_item['current_loc'] if self.pad_item is not
            None else None,
            'tim_pad': self.pad_item['current_tim'] if self.pad_item is not
            None else None,
            'text_pad': self.pad_item['text'] if self.pad_item is not
            None else None,
            'word_vec': self.data['word_vec'],
            'text_size': self.data['text_size']
        }
        return res

    def cutter_filter(self):
        """
        切割后的轨迹存储格式: (dict)
        还需要考虑语义信息，将每个点对应的语义信息加入进去
        """
        """
            {
                uid: [
                    [
                        [loc, tim, [useful word list]],
                        [loc, tim, [useful word list]],
                        ...
                    ],
                    [
                        [loc, tim, [useful word list]],
                        [loc, tim, [useful word list]],
                        ...
                    ],
                    ...
                ],
                ...
            }
        """
        # load data according to config
        traj = pd.read_csv(os.path.join(
            self.data_path, '{}.dyna'.format(self.config['dataset'])))
        poi = pd.read_csv(os.path.join(
            self.data_path, '{}.geo'.format(self.config['dataset'])))
        # 统计语料库中出现在轨迹数据集中的单词
        useful_vec = {}
        text_vec = self.load_wordvec()  # 加载语料库
        user_set = pd.unique(traj['entity_id'])
        res = {}
        min_session_len = self.config['min_session_len']
        min_sessions = self.config['min_sessions']
        time_window_size = 24  # serm 论文的时间编码方式比较独特
        base_zero = time_window_size > 12
        useful_uid = 0  # 因为有些用户会被我们删除掉，所以需要对 uid 进行重新编号
        useful_loc = {}  # loc 同理
        loc_id = 0
        for uid in user_set:
            usr_traj = traj[traj['entity_id'] == uid]
            sessions = []  # 存放该用户所有的 session
            session = []  # 单条轨迹
            # 这里还是使用当地时间吧
            start_time = parse_time(usr_traj.iloc[0]['time'], int(
                usr_traj.iloc[0]['timezone_offset_in_minutes']))
            base_time = cal_basetime(start_time, base_zero)
            for index, row in usr_traj.iterrows():
                if index == 0:
                    assert start_time.hour - base_time.hour < time_window_size
                    # 处理第一个点的语义信息
                    useful_words_list = []
                    if self.config['dataset'] in ['foursquare_tky',
                                                  'foursquare_nyk']:
                        # TODO: 这种硬编码可能不太好
                        words = poi.iloc[row['location']
                                         ]['venue_category_name'].split(' ')
                        for w in words:
                            w = w.lower()
                            if (w in text_vec) and (w not in useful_vec):
                                useful_vec[w] = text_vec[w]
                            if w in useful_vec:
                                useful_words_list.append(w)
                    time_code = start_time.hour - base_time.hour
                    if start_time.weekday() == 5 or start_time.weekday() == 6:
                        time_code += 24
                    session.append([row['location'], time_code,
                                    useful_words_list])
                else:
                    now_time = parse_time(row['time'], int(
                        row['timezone_offset_in_minutes']))
                    time_off = cal_timeoff(now_time, base_time)
                    # 处理语义
                    useful_words_list = []
                    if self.config['dataset'] in ['foursquare_tky',
                                                  'foursquare_nyk']:
                        # TODO: 这种硬编码可能不太好
                        words = poi.iloc[row['location']
                                         ]['venue_category_name'].split(' ')
                        for w in words:
                            w = w.lower()
                            if (w in text_vec) and (w not in useful_vec):
                                useful_vec[w] = text_vec[w]
                            if w in useful_vec:
                                useful_words_list.append(w)
                    if time_off < time_window_size and time_off >= 0:
                        # 特殊的时间编码
                        time_code = int(time_off)
                        if now_time.weekday() in [5, 6]:
                            time_code += 24
                        assert int(time_off) < time_window_size
                        session.append([row['location'], time_code,
                                        useful_words_list])
                    else:
                        if len(session) >= min_session_len:
                            sessions.append(session)
                        session = []
                        start_time = now_time
                        base_time = cal_basetime(start_time, base_zero)
                        time_code = start_time.hour - base_time.hour
                        if start_time.weekday() in [5, 6]:
                            time_code += 24
                        session.append([row['location'], time_code,
                                        useful_words_list])
            if len(session) >= min_session_len:
                sessions.append(session)
            if len(sessions) >= min_sessions:
                # 到这里才确定 sessions 里的 loc 都是会被使用到的
                for i in range(len(sessions)):
                    for j in range(len(sessions[i])):
                        loc = sessions[i][j][0]
                        if loc not in useful_loc:
                            useful_loc[loc] = loc_id
                            loc_id += 1
                        sessions[i][j][0] = useful_loc[loc]
                res[useful_uid] = sessions
                useful_uid += 1
        # 这里的 uid_size 和 loc_size 可能要大于实际的 uid 和 loc，因为有些可能被过滤掉了
        loc_size = loc_id
        uid_size = useful_uid
        # 根据 useful_vec 计算 word_vec
        word_index = {}
        word_vec = []
        text_size = len(useful_vec)
        for i, w in enumerate(useful_vec.keys()):
            word_index[w] = i
            word_vec.append(useful_vec[w])
        print('loc_size: {}, uid_size: {}, text_size: {}'.format(
            loc_size, uid_size, text_size))
        return {
            'loc_size': loc_size,
            'tim_size': 48,
            'uid_size': uid_size,
            'text_size': text_size,
            'word_vec': word_vec,
            'word_index': word_index,
            'data': res
        }

    def gen_input(self):
        """

        Returns:
            tuple: tuple contains:
                train_data (list)
                eval_data (list)
                test_data (list)
        """
        train_data = []
        eval_data = []
        test_data = []
        train_rate = self.config['train_rate']
        eval_rate = self.config['eval_rate']
        user_set = self.data['data'].keys()
        # 语义需要将 useful_word_list 转换成对应 word 索引的数组，再转成独热码数组，再相加归一化
        word_index = self.data['word_index']
        text_size = self.data['text_size']
        word_one_hot_matrix = np.eye(text_size)
        for u in user_set:
            sessions = self.data['data'][u]
            sessions_len = len(sessions)
            # 根据 sessions_len 来划分 train eval test
            train_num = math.ceil(sessions_len * train_rate)
            eval_num = math.ceil(
                sessions_len * (train_rate + eval_rate))
            history_session = []
            for i in range(sessions_len):
                trace = []
                if i == 0:
                    history_session += sessions[i]
                    continue
                current_session = sessions[i]
                if len(current_session) <= 1:
                    continue
                # 为当前轨迹的最后一个点 target
                target = current_session[-1][0]
                target_tim = current_session[-1][1]
                # 把多个 history 路径合并成一个？
                history_loc = [s[0]
                               for s in history_session]
                history_tim = [s[1] for s in history_session]
                trace.append(history_loc)
                trace.append(history_tim)
                loc_tim = []
                loc_tim.extend([(s[0], s[1]) for s in current_session[:-1]])
                loc_np = [s[0] for s in loc_tim]
                tim_np = [s[1] for s in loc_tim]
                # 处理语义信息
                current_word_list = [s[2] for s in current_session[:-1]]
                current_word_vec = []
                for words in current_word_list:
                    words_index = []
                    for w in words:
                        words_index.append(word_index[w])
                    if len(words_index) == 0:
                        words_vec = np.zeros((text_size))  # 没有有用的就是 0
                    else:
                        words_vec = np.sum(
                            word_one_hot_matrix[words_index], axis=0) / len(
                            words_index)
                    current_word_vec.append(words_vec)
                # loc 会与 history loc 有重合， loc 的前半部分为 history loc
                trace.append(loc_np)
                trace.append(tim_np)
                trace.append(target)
                trace.append(target_tim)
                trace.append(int(u))
                trace.append(current_word_vec)
                if i <= train_num:
                    train_data.append(trace)
                elif i <= eval_num:
                    eval_data.append(trace)
                else:
                    test_data.append(trace)
                # 将当前轨迹加入历史轨迹中
                history_session += current_session
        return train_data, eval_data, test_data

    def load_wordvec(self, vecpath=WORD_VEC_PATH):
        word_vec = {}
        with open(vecpath, 'r', encoding='utf-8') as f:
            for l in f:
                vec = []
                attrs = l.replace('\n', '').split(' ')
                for i in range(1, len(attrs)):
                    vec.append(float(attrs[i]))
                word_vec[attrs[0]] = vec
        return word_vec
