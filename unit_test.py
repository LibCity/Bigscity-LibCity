import torch
from trafficdl.config import ConfigParser
from trafficdl.data import get_dataset
from trafficdl.utils import get_model

#############################################
# The parameter to control the unit testing #
tested_trajectory_model = 'TemplateTLP'
tested_trajectory_dataset = 'foursquare_tky'
tested_trajectory_encoder = 'StandardTrajectoryEncoder'
tested_traffic_state_model = 'DCRNN'
tested_traffic_state_dataset = 'METR_LA'
#############################################


def test_new_tlp_model():
    # load Config Module
    config = ConfigParser(task='traj_loc_pred', model=tested_trajectory_model, dataset=tested_trajectory_dataset,
                          config_file=None, other_args={'batch_size': 2})
    # load Data Module
    dataset = get_dataset(config)
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()
    # get a batch to test model API
    batch = train_data.__iter__().__next__()
    batch.to_tensor(config['device'])
    # init model
    model = get_model(config, data_feature)
    model = model.to(config['device'])
    # test model.predict
    res = model.predict(batch)
    # check res format
    assert torch.is_tensor(res)
    assert res.shape[0] == batch['target'].shape[0]
    assert res.shape[1] == data_feature['loc_size']
    # test model.calculate_loss
    loss = model.calculate_loss(batch)
    assert loss.requires_grad


def test_new_tsp_model():
    # load Config Module
    config = ConfigParser(task='traffic_state_pred', model=tested_traffic_state_model,
                          dataset=tested_traffic_state_dataset,
                          config_file=None, other_args={'batch_size': 2})
    # load Data Module
    dataset = get_dataset(config)
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()
    # get a batch to test model API
    batch = train_data.__iter__().__next__()
    batch.to_tensor(config['device'])
    # init model
    model = get_model(config, data_feature)
    model = model.to(config['device'])
    # test model.predict
    res = model.predict(batch)
    assert torch.is_tensor(res)
    # test model.calculate_loss
    loss = model.calculate_loss(batch)
    assert torch.is_tensor(loss)


def test_new_traj_encoder():
    # load Config Module
    config = ConfigParser(task='traj_loc_pred', model=tested_trajectory_model, dataset=tested_trajectory_dataset,
                          config_file=None, other_args={'batch_size': 2, 'traj_encoder': tested_trajectory_encoder})
    # load dataset
    dataset = get_dataset(config)
    # cut trajectory
    cut_data = dataset.cutter_filter()
    # test encoder.encode
    encoder = dataset.encoder
    uid = list(cut_data.keys())[0]
    encoded_traj = encoder.encode(uid, cut_data[uid])
    assert isinstance(encoded_traj, list)
    an_model_input = encoded_traj[0]
    assert isinstance(an_model_input, list)
    assert len(an_model_input) == len(encoder.feature_dict)
