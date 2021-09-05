from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_model

config = ConfigParser('traj_loc_pred', 'STRNN', 'foursquare_tky', other_args={"gpu": False})
dataset = get_dataset(config)
train_data, valid_data, test_data = dataset.get_data()
data_feature = dataset.get_data_feature()
# batch = valid_data.__iter__().__next__()
model = get_model(config, data_feature)
self = model.to(config['device'])
# batch.to_tensor(config['device'])
'''
user = batch['uid']
dst = batch['target'].tolist()
dst_time = batch['target_tim']
current_loc = batch['current_loc']
current_tim = batch['current_tim']
# 计算 td ld
batch_size = len(dst)
td = dst_time.unsqueeze(1) - current_tim
ld = torch.zeros(current_loc.shape).to(self.device)
loc_len = batch.get_origin_len('current_loc')
current_loc = current_loc.tolist()
for i in range(batch_size):
    target = dst[i]
    lon_i, lat_i = parseCoordinate(self.poi_profile.iloc[target]['coordinates'])
    for j in range(loc_len[i]):
        origin = current_loc[i][j]
        lon_j, lat_j = parseCoordinate(self.poi_profile.iloc[origin]['coordinates'])
        # 计算 target - origin 的距离，并写入 ld[i][j] 中
        ld[i][j] = distance.distance((lat_i, lon_i), (lat_j, lon_j)).kilometers

td_upper = torch.LongTensor([self.up_time] * batch_size).to(self.device).unsqueeze(1)
td_upper = td_upper - td
td_lower = td # 因为 lower 是 0
ld_upper = torch.LongTensor([self.up_loc] * batch_size).to(self.device).unsqueeze(1)
ld_upper = ld_upper - ld
ld_lower = ld # 因为下界是 0

for idx, batch in enumerate(train_data):
    batch.to_tensor(device=config['device'])
    current_loc = batch['current_loc'].tolist()
    batch_size = len(current_loc)
    loc_len = batch.get_origin_len('current_loc')
    for i in range(batch_size):
            for j in range(loc_len[i]):
                    if current_loc[i][j] >= 94890 or current_loc[i][j] < 0:
                        print('index error')
                        break

'''
