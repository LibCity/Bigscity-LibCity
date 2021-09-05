from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_model, get_evaluator
# load config
config = ConfigParser('traj_loc_pred', 'ATSTLSTM', 'foursquare_tky', other_args={'min_sessions': 5})

dataset = get_dataset(config)
train_data, valid_data, test_data = dataset.get_data()
data_feature = dataset.get_data_feature()

model = get_model(config, data_feature).to(config['device'])
batch = train_data.__iter__().__next__()
batch.to_tensor(config['device'])

score = model.predict(batch)
loc_true = [0] * config['batch_size']
evaluate_input = {
    'uid': batch['uid'].tolist(),
    'loc_true': loc_true,
    'loc_pred': score.tolist()
}

evaluator = get_evaluator(config)
evaluator.collect(evaluate_input)
