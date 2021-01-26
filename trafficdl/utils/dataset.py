'''
数据预处理阶段相关的工具函数
'''
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
'''
将 json 中 time_format 格式的 time 转化为 local datatime
'''
def parseTime(time, timezone_offset_in_minute):
    '''
    parse to datetime
    '''
    date = datetime.strptime(time, '%Y-%m-%dT%H:%M:%SZ') # 这是 UTC 时间
    return date + timedelta(minutes=timezone_offset_in_minute)

'''
用于切分轨迹成一个 session
思路为：给定一个 start_time 找到一个基准时间 base_time，在该 base_time 到 base_time + time_length 区间的点划分到一个 session 内
选取 base_time 来做的理由是：这样可以保证同一个小时段总是被 encode 成同一个数
'''
def calculateBaseTime(start_time, base_zero):
    if base_zero:
        return start_time - timedelta(hours=start_time.hour, minutes=start_time.minute, seconds=start_time.second,microseconds=start_time.microsecond)
    else:
        # time length = 12
        if start_time.hour < 12:
            return start_time - timedelta(hours=start_time.hour, minutes=start_time.minute, seconds=start_time.second,microseconds=start_time.microsecond)
        else:
            return start_time - timedelta(hours=start_time.hour - 12, minutes=start_time.minute, seconds=start_time.second,microseconds=start_time.microsecond)

'''
计算两个时间之间的差值，返回值以小时为单位
'''
def calculateTimeOff(now_time, base_time):
    # 先将 now 按小时对齐
    now_time = now_time - timedelta(minutes=now_time.minute, seconds=now_time.second)
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
    sim_matrix = np.zeros((tim_size,tim_size))
    for i in range(tim_size):
        for j in range(tim_size):
            set_i = time_checkin_set[i]
            set_j = time_checkin_set[j]
            if len(set_i | set_j) != 0:
                jaccard_ij = len(set_i & set_j)/len(set_i | set_j)
                sim_matrix[i][j] = jaccard_ij
    return sim_matrix

def parseCoordinate(coordinate):
    items = coordinate[1:-1].split(',')
    return float(items[0]), float(items[1])
