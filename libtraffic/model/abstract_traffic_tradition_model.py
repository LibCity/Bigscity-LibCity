class AbstractTraditionModel:

    def __init__(self, config, data_feature):
        self.data_feature = data_feature

    def run(self, data):
        """
        Args:
            data : input of tradition model

        Returns:
            output of tradition model
        """
