"""
数据预处理阶段相关的工具函数
"""
import numpy as np
import time
from datetime import datetime, timedelta
from collections import defaultdict


def parse_time(time_in, timezone_offset_in_minute=0):
    """
    将 json 中 time_format 格式的 time 转化为 local datatime
    """
    date = datetime.strptime(time_in, '%Y-%m-%dT%H:%M:%SZ')  # 这是 UTC 时间
    return date + timedelta(minutes=timezone_offset_in_minute)


def cal_basetime(start_time, base_zero):
    """
    用于切分轨迹成一个 session，
    思路为：给定一个 start_time 找到一个基准时间 base_time，
    在该 base_time 到 base_time + time_length 区间的点划分到一个 session 内，
    选取 base_time 来做的理由是：这样可以保证同一个小时段总是被 encode 成同一个数
    """
    if base_zero:
        return start_time - timedelta(hours=start_time.hour,
                                      minutes=start_time.minute,
                                      seconds=start_time.second,
                                      microseconds=start_time.microsecond)
    else:
        # time length = 12
        if start_time.hour < 12:
            return start_time - timedelta(hours=start_time.hour,
                                          minutes=start_time.minute,
                                          seconds=start_time.second,
                                          microseconds=start_time.microsecond)
        else:
            return start_time - timedelta(hours=start_time.hour - 12,
                                          minutes=start_time.minute,
                                          seconds=start_time.second,
                                          microseconds=start_time.microsecond)


def cal_timeoff(now_time, base_time):
    """
    计算两个时间之间的差值，返回值以小时为单位
    """
    # 先将 now 按小时对齐
    delta = now_time - base_time
    return delta.days * 24 + delta.seconds / 3600


def caculate_time_sim(data):
    time_checkin_set = defaultdict(set)
    tim_size = data['tim_size']
    data_neural = data['data']
    for uid in data_neural:
        uid_sessions = data_neural[uid]
        for session in uid_sessions:
            for checkin in session:
                timid = checkin[1]
                locid = checkin[0]
                if timid not in time_checkin_set:
                    time_checkin_set[timid] = set()
                time_checkin_set[timid].add(locid)
    sim_matrix = np.zeros((tim_size, tim_size))
    for i in range(tim_size):
        for j in range(tim_size):
            set_i = time_checkin_set[i]
            set_j = time_checkin_set[j]
            if len(set_i | set_j) != 0:
                jaccard_ij = len(set_i & set_j) / len(set_i | set_j)
                sim_matrix[i][j] = jaccard_ij
    return sim_matrix


def parse_coordinate(coordinate):
    items = coordinate[1:-1].split(',')
    return float(items[0]), float(items[1])


def string2timestamp(strings, offset_frame):
    ts = []
    for t in strings:
        dtstr = '-'.join([t[:4].decode(), t[4:6].decode(), t[6:8].decode()])
        slot = int(t[8:]) - 1
        ts.append(np.datetime64(dtstr, 'm') + slot * offset_frame)
    return ts  # [numpy.datetime64('2014-01-01T00:00'), ...]


def timestamp2array(timestamps, t):
    """
    把时间戳的序列中的每一个时间戳转成特征数组，考虑了星期和小时，
    时间戳: numpy.datetime64('2013-07-01T00:00:00.000000000')

    Args:
        timestamps: 时间戳序列
        t: 一天有多少个时间步

    Returns:
        np.ndarray: 特征数组，shape: (len(timestamps), ext_dim)
    """
    vec_wday = [time.strptime(
        str(t)[:10], '%Y-%m-%d').tm_wday for t in timestamps]
    vec_hour = [time.strptime(str(t)[11:13], '%H').tm_hour for t in timestamps]
    vec_minu = [time.strptime(str(t)[14:16], '%M').tm_min for t in timestamps]
    ret = []
    for idx, wday in enumerate(vec_wday):
        # day
        v = [0 for _ in range(7)]
        v[wday] = 1
        if wday >= 5:  # 0是周一, 6是周日
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday len(v)=8
        # hour
        v += [0 for _ in range(t)]  # len(v)=8+T
        hour = vec_hour[idx]
        minu = vec_minu[idx]
        # 24*60/T 表示一个时间步是多少分钟
        # hour * 60 + minu 是从0:0开始到现在是多少分钟，相除计算是第几个时间步
        # print(hour, minu, T, (hour * 60 + minu) / (24 * 60 / T))
        v[int((hour * 60 + minu) / (24 * 60 / t))] = 1
        # +8是因为v前边有表示星期的8位
        if hour >= 18 or hour < 6:
            v.append(0)  # night
        else:
            v.append(1)  # day
        ret.append(v)  # len(v)=7+1+T+1=T+9
    return np.asarray(ret)


def timestamp2vec_origin(timestamps):
    """
    把时间戳的序列中的每一个时间戳转成特征数组，只考虑星期，
    时间戳: numpy.datetime64('2013-07-01T00:00:00.000000000')

    Args:
        timestamps: 时间戳序列

    Returns:
        np.ndarray: 特征数组，shape: (len(timestamps), 8)
    """
    vec = [time.strptime(str(t)[:10], '%Y-%m-%d').tm_wday for t in timestamps]
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        ret.append(v)
    return np.asarray(ret)
