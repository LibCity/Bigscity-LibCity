'''
数据预处理阶段相关的工具函数
'''
from datetime import datetime, timedelta

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
