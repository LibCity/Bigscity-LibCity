
class ConfigParser(object):
    '''
    use to parse the user defined parameters and use these to modify the pipeline's parameter setting.
    '''
    
    def __init__(self, task, model, dataset, config_file, other_args):
        self.config = {}
    
