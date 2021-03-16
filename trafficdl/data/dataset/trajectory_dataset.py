import os
import json
import pandas as pd
import math

from trafficdl.data.dataset import AbstractDataset
from trafficdl.utils import parse_time, cal_basetime, cal_timeoff, \
    caculate_time_sim
from trafficdl.data.utils import generate_dataloader

parameter_list = ['dataset', 'min_session_len', 'min_sessions',
                  'time_window_size']


class TrajectoryDataset(AbstractDataset):

    def __init__(self, config):
        self.config = config
        parameters_str = ''
        for key in parameter_list:
            if key in self.config:
                parameters_str += '_' + str(self.config[key])
        self.cache_file_name = os.path.join(
            './trafficdl/cache/dataset_cache/', 'trajectory_{}.json'
            .format(parameters_str))  # 缓存切好的轨迹
        self.cache_file_folder = './trafficdl/cache/dataset_cache/'
        self.data_path = './raw_data/{}/'.format(self.config['dataset'])
        self.data = None
        self.pad_item = None
        self.pad_max_len = {
            'history_loc': self.config['history_len'],
            'history_tim': self.config['history_len']
        }
        self.feature_name = {'history_loc': 'int', 'history_tim': 'int',
                             'current_loc': 'int', 'current_tim': 'int',
                             'target': 'int', 'target_tim': 'int', 'uid': 'int'
                             }
        self.history_type = config['history_type']
        # 对于这种 history 模式没办法做到 batch
        if self.history_type == 'cut_off':
            # self.config['batch_size'] = 1
            self.feature_name['history_loc'] = 'array of int'
            self.feature_name['history_tim'] = 'array of int'

    def get_data(self):
        """
        轨迹比较特殊，原子文件中存储的并不是轨迹而是一个一个点，因此需要先对轨迹进行切割
        """
        if self.data is None:
            if self.config['cache_dataset'] and \
                    os.path.exists(self.cache_file_name):
                # load cache
                f = open(self.cache_file_name, 'r')
                self.data = json.load(f)
                loc_pad = self.data['loc_size'] - 1
                tim_pad = self.data['tim_size'] - 1
                if self.history_type == 'cut_off':
                    self.pad_item = {
                        'current_loc': loc_pad,
                        'current_tim': tim_pad
                    }
                    # 这种情况下不对 history_loc history_tim 做补齐
                else:
                    self.pad_item = {
                        'current_loc': loc_pad,
                        'history_loc': loc_pad,
                        'current_tim': tim_pad,
                        'history_tim': tim_pad
                    }
                f.close()
            else:
                transformed_data = self.cutter_filter()
                # pad parameter
                loc_pad = transformed_data['loc_size']
                transformed_data['loc_size'] += 1
                tim_pad = transformed_data['tim_size']
                transformed_data['tim_size'] += 1
                if self.history_type == 'cut_off':
                    self.pad_item = {
                        'current_loc': loc_pad,
                        'current_tim': tim_pad
                    }
                    # 这种情况下不对 history_loc history_tim 做补齐
                else:
                    self.pad_item = {
                        'current_loc': loc_pad,
                        'history_loc': loc_pad,
                        'current_tim': tim_pad,
                        'history_tim': tim_pad
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
        return generate_dataloader(train_data, eval_data, test_data,
                                   self.feature_name,
                                   self.config['batch_size'],
                                   self.config['num_workers'], self.pad_item,
                                   self.pad_max_len)

    def get_data_feature(self):
        res = {
            'loc_size': self.data['loc_size'],
            'tim_size': self.data['tim_size'],
            'uid_size': self.data['uid_size'],
            'loc_pad': self.pad_item['current_loc'] if self.pad_item is not
            None else None,
            'tim_pad': self.pad_item['current_tim'] if self.pad_item is not
            None else None
        }
        res['poi_profile'] = pd.read_csv(os.path.join(
            self.data_path, '{}.geo'.format(self.config['dataset'])))
        with open(os.path.join(self.data_path, 'config.json'), 'r') as f:
            config = json.load(f)
            res['distance_upper'] = config['extra']['distance_upper']
        if self.config['model'] == 'LSTPM':
            # 这个模型需要加一个新的东西
            res['tim_sim_matrix'] = caculate_time_sim(self.data)
        return res

    def cutter_filter(self):
        """
        切割后的轨迹存储格式: (dict)
            {
                uid: [
                    [
                        [loc, tim],
                        [loc, tim],
                        ...
                    ],
                    [
                        [loc, tim],
                        [loc, tim],
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
        user_set = pd.unique(traj['entity_id'])
        res = {}
        min_session_len = self.config['min_session_len']
        min_sessions = self.config['min_sessions']
        time_window_size = self.config['time_window_size']
        base_zero = time_window_size > 12
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
                    # time encode from 0 ~ time_window_size
                    session.append(
                        [row['location'], start_time.hour - base_time.hour])
                else:
                    now_time = parse_time(row['time'], int(
                        row['timezone_offset_in_minutes']))
                    time_off = cal_timeoff(now_time, base_time)
                    if time_off < time_window_size and time_off >= 0:
                        assert int(time_off) < time_window_size
                        session.append([row['location'], int(time_off)])
                    else:
                        if len(session) >= min_session_len:
                            sessions.append(session)
                        session = []
                        start_time = now_time
                        base_time = cal_basetime(start_time, base_zero)
                        session.append(
                            [row['location'], start_time.hour - base_time.hour]
                            )
            if len(session) >= min_session_len:
                sessions.append(session)
            if len(sessions) >= min_sessions:
                res[str(uid)] = sessions
        # 这里的 uid_size 和 loc_size 可能要大于实际的 uid 和 loc，因为有些可能被过滤掉了
        poi = pd.read_csv(os.path.join(
            self.data_path, '{}.geo'.format(self.config['dataset'])))
        loc_size = poi.shape[0]
        uid_size = len(user_set)
        print('loc_size: {}, uid_size: {}'.format(loc_size, uid_size))
        return {
            'loc_size': loc_size,
            'tim_size': time_window_size,
            'uid_size': uid_size,
            'data': res
        }

    def gen_input(self):
        """
        return:
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
                    if self.history_type == 'splice':
                        history_session += sessions[i]
                    else:
                        history_session.append(sessions[i])
                    continue
                current_session = sessions[i]
                if len(current_session) <= 1:
                    continue
                # 为当前轨迹的最后一个点 target
                target = current_session[-1][0]
                target_tim = current_session[-1][1]
                if self.history_type == 'splice':
                    # 把多个 history 路径合并成一个？
                    history_loc = [s[0] for s in history_session]
                    history_tim = [s[1] for s in history_session]
                else:
                    history_loc = []
                    history_tim = []
                    for s in history_session:
                        temp_loc = [ss[0] for ss in s]
                        temp_tim = [ss[1] for ss in s]
                        history_loc.append(temp_loc)
                        history_tim.append(temp_tim)
                trace.append(history_loc)
                trace.append(history_tim)
                loc_tim = []
                loc_tim.extend([(s[0], s[1]) for s in current_session[:-1]])
                loc_np = [s[0] for s in loc_tim]
                tim_np = [s[1] for s in loc_tim]
                # loc 会与 history loc 有重合， loc 的前半部分为 history loc
                trace.append(loc_np)
                trace.append(tim_np)
                trace.append(target)
                trace.append(target_tim)
                trace.append(int(u))
                if i <= train_num:
                    train_data.append(trace)
                elif i <= eval_num:
                    eval_data.append(trace)
                else:
                    test_data.append(trace)
                # 将当前轨迹加入历史轨迹中
                if self.history_type == 'splice':
                    history_session += current_session
                else:
                    history_session.append(current_session)
        return train_data, eval_data, test_data
