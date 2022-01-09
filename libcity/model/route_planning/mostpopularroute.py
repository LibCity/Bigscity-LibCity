from libcity.model.abstract_machine_learning_model import AbstractMachineLearningModel
from libcity.utils.transfer_probability import TransferProbability

class MPR(AbstractMachineLearningModel):
    def __init__(self, config, data_feature):
        """

        Args:
            config (ConfigParser): the dict of config
            data_feature (dict): the dict of data feature passed from Dataset.
        """
        super().__init__(config, data_feature)
        self.data_feature = data_feature
        self.road_gps = self.data_feature.get('road_gps')
        self.road_num = self.data_feature.get('loc_num')
        self.traj_num = self.data_feature.get('traj_num')
        self.adj_mx = self.data_feature.get('adj_mx')

        self.transferpro_mx = None

    def predict(self, batch):
        """
        Args:
            batch (Batch): a batch of input

        Returns:
            result (object) : predict result of this batch
        """

    def fit(self, batch):
        """
        use train data to fit the machine learning model

        Args:
            batch (Batch): a batch of input. Generally speaking, the train data of machine learning method
            does not need to be divided into batches, just use Batch to pass in all the data directly.

        Returns:
            return None
        """
        trajectories = dict()
        traj_set = set()
        traj_dict = {}
        for traj_id, trace in enumerate(batch):
            current_trace = batch[traj_id]
            trajectories[traj_id] = current_trace
            for step in range(len(current_trace) - 1):
                f_id = current_trace[step]
                t_id = current_trace[step + 1]
                if (f_id, t_id) not in traj_set:
                    traj_set.add((f_id, t_id))
                    traj_dict[(f_id, t_id)] = [traj_id]
                else:
                    if traj_id not in traj_dict[(f_id, t_id)]:
                        traj_dict[(f_id, t_id)].append(traj_id)
                    else:
                        pass
        self.transferpro_mx = TransferProbability(self.road_gps, traj_dict, trajectories)
