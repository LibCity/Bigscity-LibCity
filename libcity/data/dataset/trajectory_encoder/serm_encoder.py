import os
import pandas as pd
from libcity.data.dataset.trajectory_encoder.abstract_trajectory_encoder import AbstractTrajectoryEncoder
from libcity.utils import parse_time

parameter_list = ['dataset', 'min_session_len', 'min_sessions', 'traj_encoder', 'cut_method',
                  'window_size', 'history_type', 'min_checkins', 'max_session_len']
WORD_VEC_PATH = './raw_data/word_vec/glove.twitter.27B.50d.txt'


class SermEncoder(AbstractTrajectoryEncoder):

    def __init__(self, config):
        super().__init__(config)
        self.uid = 0
        self.location2id = {}  # 因为原始数据集中的部分 loc id 不会被使用到因此这里需要重新编码一下
        self.loc_id = 0
        self.tim_max = 47  # 时间编码方式得改变
        self.word_vec = []  # words vector
        self.word_index = {}  # word to word ID
        self.word_id = 0
        self.text_vec = self.load_wordvec()
        self.history_type = self.config['history_type']
        self.feature_dict = {'current_loc': 'int', 'current_tim': 'int',
                             'target': 'int', 'uid': 'int', 'text': 'no_tensor'}
        # if config['evaluate_method'] == 'sample':
        #     self.feature_dict['neg_loc'] = 'int'
        #     parameter_list.append('neg_samples')
        parameters_str = ''
        for key in parameter_list:
            if key in self.config:
                parameters_str += '_' + str(self.config[key])
        self.cache_file_name = os.path.join(
            './libcity/cache/dataset_cache/', 'trajectory_{}.json'.format(parameters_str))
        # load poi_profile
        self.poi_profile = None
        self.dataset = self.config.get('dataset', '')
        self.geo_file = self.config.get('geo_file', self.dataset)
        if self.dataset in ['foursquare_tky', 'foursquare_nyk', 'foursquare_serm']:
            self.poi_profile = pd.read_csv('./raw_data/{}/{}.geo'.format(self.dataset,
                                                                         self.geo_file))

    def encode(self, uid, trajectories, negative_sample=None):
        """standard encoder use the same method as DeepMove

        Recode poi id. Encode timestamp with its hour.

        Args:
            uid ([type]): same as AbstractTrajectoryEncoder
            trajectories ([type]): same as AbstractTrajectoryEncoder
                trajectory1 = [
                    (location ID, timestamp, timezone_offset_in_minutes),
                    (location ID, timestamp, timezone_offset_in_minutes),
                    .....
                ]
        """
        # 直接对 uid 进行重编码
        uid = self.uid
        self.uid += 1
        encoded_trajectories = []
        for index, traj in enumerate(trajectories):
            current_loc = []
            current_tim = []
            current_word_vec = []
            for point in traj:
                loc = point[4]
                now_time = parse_time(point[2])
                if loc not in self.location2id:
                    self.location2id[loc] = self.loc_id
                    self.loc_id += 1
                current_loc.append(self.location2id[loc])
                # 采用工作日编码到0-23，休息日编码到24-47
                time_code = self._time_encode(now_time)
                current_tim.append(time_code)
                # 处理语义信息
                current_word_vec.append(self.get_text_from_point(point))
            # 完成当前轨迹的编码，下面进行输入的形成
            # 一条轨迹可以产生多条训练数据，根据第一个点预测第二个点，前两个点预测第三个点....
            for i in range(len(current_loc) - 1):
                trace = []
                target = current_loc[i+1]
                trace.append(current_loc[:i+1])
                trace.append(current_tim[:i+1])
                trace.append(target)
                trace.append(uid)
                trace.append(current_word_vec[:i+1])
                # if negative_sample is not None:
                #     neg_loc = []
                #     for neg in negative_sample[index]:
                #         if neg not in self.location2id:
                #             self.location2id[neg] = self.loc_id
                #             self.loc_id += 1
                #         neg_loc.append(self.location2id[neg])
                #     trace.append(neg_loc)
                encoded_trajectories.append(trace)
        return encoded_trajectories

    def gen_data_feature(self):
        loc_pad = self.loc_id
        tim_pad = self.tim_max + 1
        self.pad_item = {
            'current_loc': loc_pad,
            'current_tim': tim_pad
        }
        self.data_feature = {
            'loc_size': self.loc_id + 1,
            'tim_size': self.tim_max + 2,
            'uid_size': self.uid,
            'loc_pad': loc_pad,
            'tim_pad': tim_pad,
            'text_size': len(self.word_index),
            'word_vec': self.word_vec
        }

    def _time_encode(self, time):
        if time.weekday() in [0, 1, 2, 3, 4]:
            return time.hour
        else:
            return time.hour + 24

    def load_wordvec(self, vecpath=WORD_VEC_PATH):
        word_vec = {}
        if not os.path.exists(vecpath):
            raise FileNotFoundError('SERM need Glove word vectors. Please download serm_glove_word_vec.zip from'
                                    ' BaiduDisk or Google Drive, and unzip it to raw_data directory')
        with open(vecpath, 'r', encoding='utf-8') as f:
            for l in f:
                vec = []
                attrs = l.replace('\n', '').split(' ')
                for i in range(1, len(attrs)):
                    vec.append(float(attrs[i]))
                word_vec[attrs[0]] = vec
        return word_vec

    def get_text_from_point(self, point):
        """
            return word index
        """
        if self.dataset in ['foursquare_tky', 'foursquare_nyc']:
            # 语义信息在 geo 表中
            words = self.poi_profile.iloc[point[4]]['venue_category_name'].split(' ')
            word_index = []
            for w in words:
                w = w.lower()
                if (w in self.text_vec) and (w not in self.word_index):
                    self.word_index[w] = self.word_id
                    self.word_id += 1
                    self.word_vec.append(self.text_vec[w])
                if w in self.word_index:
                    word_index.append(self.word_index[w])
            return word_index
        else:
            raise TypeError('SERM model can only run on foursquare dataset, because it needs POI category information.')
