import datetime

import numpy as np
import pandas as pd


from libcity.data.dataset import TrafficStateCPTDataset


class GSNetDataset(TrafficStateCPTDataset):
    def __init__(self, config):
        # initialize here for calling self._load_rel() properly
        self.weight_col_list = config.get('weight_col', [])

        super(GSNetDataset, self).__init__(config)

        # for properly loading the dyna file
        self.len_row = config.get('grid_len_row', None)
        self.len_column = config.get('grid_len_column', None)

        # NOTE: DOES NOT take into account time-of-day and day-of-week rows
        self.num_of_target_time_feature = self.config.get('num_of_target_time_feature', 0)
        self.grid_in_channel = len(self.config.get('data_col', []))
        if self.add_time_in_day:
            self.num_of_target_time_feature += 24
            self.grid_in_channel += 24
        if self.add_day_in_week:
            self.num_of_target_time_feature += 7
            self.grid_in_channel += 7

        self.data_col_risk_mask = self.config.get('data_col_risk_mask', 'risk_mask')
        self.data_col_grid_node_map = self.config.get('data_col_grid_node_map', 'grid_node_map')

        self._load_risk_mask(self.dataset)

    def _load_rel(self):
        try:
            orig_weight_col = self.weight_col
        except AttributeError:
            orig_weight_col = None
        try:
            orig_adj_mx = self.adj_mx
        except AttributeError:
            orig_adj_mx = None
        try:
            orig_distance_df = self.distance_df
        except AttributeError:
            orig_distance_df = None

        # code reuse
        try:
            self.weight_col = self.weight_col_list[0]
            super(GSNetDataset, self)._load_rel()
            self.road_adj = self.adj_mx

            self.weight_col = self.weight_col_list[1]
            super(GSNetDataset, self)._load_rel()
            self.risk_adj = self.adj_mx

            if len(self.weight_col_list) > 2:
                self.weight_col = self.weight_col_list[2]
                super(GSNetDataset, self)._load_rel()
                self.poi_adj = self.adj_mx
        finally:
            self.weight_col = orig_weight_col
            self.adj_mx = orig_adj_mx
            self.distance_df = orig_distance_df

    def _load_dyna(self, filename):
        # dynamic data must be 4D in this model
        # fake grid-based geoids
        orig_geo_ids = self.geo_ids
        self.geo_ids = [i * self.len_column + j for i in range(self.len_row) for j in range(self.len_column)]

        result = super(GSNetDataset, self)._load_grid_4d(filename)

        self.geo_ids = orig_geo_ids
        return result

    # for grid-based auxillary matrices
    def _load_risk_mask(self, filename):
        self._logger.info("Loading file " + filename + ".geo")
        df = pd.read_csv(self.data_path + filename + '.geo')
        # column first, reflecting the model's preference
        len_row, len_column = self.len_row, self.len_column
        num_graph_nodes = len(df)
        risk_mask_values = []
        k = 0
        for i in range(len_row):
            for j in range(len_column):
                if k < num_graph_nodes and i == df['row_id'][k] and j == df['column_id'][k]:
                    risk_mask_values.append(df['risk_mask'][k])
                    k += 1
                else:
                    risk_mask_values.append(0.0)
        grid_node_map_values = []
        for i in range(len_row * len_column):
            grid_node_map_values.append([0.0] * num_graph_nodes)
        for i in range(num_graph_nodes):
            index = df['row_id'][i] * len_column + df['column_id'][i]
            grid_node_map_values[index][df['geo_id'][i]] = 1.0

        self.risk_mask = np.array(risk_mask_values, dtype=np.float32).reshape(len_column, len_row)
        self.grid_node_map = np.array(grid_node_map_values, dtype=np.float32).reshape(len_column * len_row,
                                                                                      num_graph_nodes)

    def _get_external_array(self, ts, ext_data=None, previous_ext=False):
        # one-hot encoding that differs from ordinary datasets
        ts_count = len(ts)
        data_list = []

        if self.add_time_in_day:
            time_indices = ((ts - ts.astype("datetime64[D]")) / np.timedelta64(1, "h")).astype("int")
            curr = np.zeros((ts_count, 24))
            # [ts_count, 24]
            curr[np.arange(0, ts_count), time_indices] = 1
            data_list.append(curr)
        if self.add_day_in_week:
            week_indices = []
            for day in ts.astype("datetime64[D]"):
                week_indices.append(datetime.datetime.strptime(str(day), '%Y-%m-%d').weekday())
            curr = np.zeros((ts_count, 7))
            curr[np.arange(0, ts_count), week_indices] = 1
            data_list.append(curr)

        if ext_data is not None:
            indexs = []
            for ts_ in ts:
                if previous_ext:
                    ts_index = self.idx_of_ext_timesolts[ts_ - self.offset_frame]
                else:
                    ts_index = self.idx_of_ext_timesolts[ts_]
                indexs.append(ts_index)
            select_data = ext_data[indexs]
            data_list.append(select_data)

        if len(data_list) > 0:
            data = np.concatenate(data_list, axis=1)
        else:
            data = np.zeros((len(ts), 0))

        return data

    def get_data_feature(self):
        d = {
                "scaler": self.scaler,
                "num_batches": self.num_batches,
                "feature_dim": self.feature_dim,
                "ext_dim": self.ext_dim,
                "output_dim": self.output_dim,
                "len_row": self.len_row,
                "len_column": self.len_column
        }

        d['risk_mask'] = self.risk_mask
        d['road_adj'] = self.road_adj
        d['risk_adj'] = self.risk_adj
        if hasattr(self, 'poi_adj'):
            d['poi_adj'] = self.poi_adj
        else:
            d['poi_adj'] = None
        d['grid_node_map'] = self.grid_node_map

        d['num_of_target_time_feature'] = self.num_of_target_time_feature

        lp = self.len_period * (self.pad_forward_period + self.pad_back_period + 1)
        lt = self.len_trend * (self.pad_forward_trend + self.pad_back_trend + 1)

        d['len_closeness'] = self.len_closeness
        d['len_period'] = lp
        d['len_trend'] = lt

        d['add_time_in_day'] = self.add_time_in_day
        d['add_day_in_week'] = self.add_day_in_week

        # what rows should be considered in transformation from grid input to the graph one, referred by indices
        data_col = self.config.get('data_col', [])
        for k in ['graph_input', 'target_time']:
            d[f'{k}_indices'] = []
            for n in self.config.get(f'{k}_col', []):
                # let ValueErrors raise
                d[f'{k}_indices'].append(data_col.index(n))

        d['risk_thresholds'] = self.config.get('risk_thresholds', [])
        d['risk_weights'] = self.config.get('risk_weights', [])
        for k in ['risk_thresholds', 'risk_weights']:
            d[k] = self.config.get(k, [])
            if d[k] != sorted(d[k]):
                raise ValueError(f'Dataset config item {k} is not a sorted list')
        if len(d['risk_thresholds']) != len(d['risk_weights']) - 1:
            raise ValueError('Mask loss risk thresholds must be one element shorter than risk weights')

        return d
