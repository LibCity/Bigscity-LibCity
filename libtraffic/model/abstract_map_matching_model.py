
class AbstractMapMatchingModel:

    def __init__(self, config, data_feature):
        raise NotImplementedError("MapMatchingModel __init__ not implemented")

    def run(self, data):
        """
        Args:
            data : a dictionary of trajectory and rd_nwk

        Returns:
            matched roads TODO
        """
        raise NotImplementedError("MapMatchingModel __init__ not implemented")