import numpy as np
from functools import partial
from logging import getLogger
from hyperopt import fmin, tpe


def _recursiveFindNodes(root, node_type='switch'):
    from hyperopt.pyll.base import Apply
    nodes = []
    if isinstance(root, (list, tuple)):
        for node in root:
            nodes.extend(_recursiveFindNodes(node, node_type))
    elif isinstance(root, dict):
        for node in root.values():
            nodes.extend(_recursiveFindNodes(node, node_type))
    elif isinstance(root, (Apply)):
        if root.name == node_type:
            nodes.append(root)

        for node in root.pos_args:
            if node.name == node_type:
                nodes.append(node)
        for _, node in root.named_args:
            if node.name == node_type:
                nodes.append(node)
    return nodes


def _parameters(space):
    # Analyze the domain instance to find parameters
    parameters = {}
    if isinstance(space, dict):
        space = list(space.values())
    for node in _recursiveFindNodes(space, 'switch'):

        # Find the name of this parameter
        paramNode = node.pos_args[0]
        assert paramNode.name == 'hyperopt_param'
        paramName = paramNode.pos_args[0].obj

        # Find all possible choices for this parameter
        values = [literal.obj for literal in node.pos_args[1:]]
        parameters[paramName] = np.array(range(len(values)))
    return parameters


def _spacesize(space):
    # Compute the number of possible combinations
    params = _parameters(space)
    return np.prod([len(values) for values in params.values()])


class ExhaustiveSearchError(Exception):
    r""" ExhaustiveSearchError

    """
    pass


def _validate_space_exhaustive_search(space):
    from hyperopt.pyll.base import dfs, as_apply
    from hyperopt.pyll.stochastic import implicit_stochastic_symbols
    supported_stochastic_symbols = ['randint', 'quniform', 'qloguniform', 'qnormal', 'qlognormal', 'categorical']
    for node in dfs(as_apply(space)):
        if node.name in implicit_stochastic_symbols:
            if node.name not in supported_stochastic_symbols:
                raise ExhaustiveSearchError('Exhaustive search is only possible with the following stochastic symbols: '
                                            '' + ', '.join(supported_stochastic_symbols))


def exhaustive_search(new_ids, domain, trials, seed, nbMaxSucessiveFailures=1000):
    r""" This is for exhaustive search in HyperTuning.

    """
    from hyperopt import pyll
    from hyperopt.base import miscs_update_idxs_vals
    # Build a hash set for previous trials
    hashset = set([hash(frozenset([(key, value[0]) if len(value) > 0 else ((key, None))
                                   for key, value in trial['misc']['vals'].items()])) for trial in trials.trials])

    rng = np.random.RandomState(seed)
    rval = []
    for _, new_id in enumerate(new_ids):
        newSample = False
        nbSucessiveFailures = 0
        while not newSample:
            # -- sample new specs, idxs, vals
            idxs, vals = pyll.rec_eval(
                domain.s_idxs_vals,
                memo={
                    domain.s_new_ids: [new_id],
                    domain.s_rng: rng,
                })
            new_result = domain.new_result()
            new_misc = dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)
            miscs_update_idxs_vals([new_misc], idxs, vals)

            # Compare with previous hashes
            h = hash(frozenset([(key, value[0]) if len(value) > 0 else (
                (key, None)) for key, value in vals.items()]))
            if h not in hashset:
                newSample = True
            else:
                # Duplicated sample, ignore
                nbSucessiveFailures += 1

            if nbSucessiveFailures > nbMaxSucessiveFailures:
                # No more samples to produce
                return []

        rval.extend(trials.new_trial_docs([new_id],
                                          [None], [new_result], [new_misc]))
    return rval


class HyperTuning(object):
    """

    Note:
        HyperTuning is based on the hyperopt (https://github.com/hyperopt/hyperopt)

        https://github.com/hyperopt/hyperopt/issues/200
    """

    def __init__(self, objective_function, space=None, params_file=None, algo='grid_search',
                 max_evals=100, task=None, model_name=None, dataset_name=None, config_file=None,
                 saved_model=True, train=True, other_args=None):
        self.task = task
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.config_file = config_file
        self.saved_model = saved_model
        self.train = train
        self.other_args = other_args
        self._logger = getLogger()

        self.best_score = None
        self.best_params = None
        self.best_test_result = None
        self.params2result = {}

        self.objective_function = objective_function
        self.max_evals = max_evals
        if space:
            self.space = space
        elif params_file:
            self.space = self._build_space_from_file(params_file)
        else:
            raise ValueError('at least one of `space` and `params_file` is provided')

        if isinstance(algo, str):
            if algo == 'grid_search':
                self.algo = partial(exhaustive_search, nbMaxSucessiveFailures=1000)
                self.max_evals = _spacesize(self.space)
            else:
                # TODO: 其他算法
                self.algo = tpe.suggest
        else:
            self.algo = algo

    @staticmethod
    def _build_space_from_file(file):
        from hyperopt import hp
        space = {}
        with open(file, 'r') as fp:
            for line in fp:
                para_list = line.strip().split(' ')
                if len(para_list) < 3:
                    continue
                para_name, para_type, para_value = para_list[0], para_list[1], "".join(para_list[2:])
                # TODO: para_value ['[]']?
                if para_type == 'choice':
                    para_value = eval(para_value)
                    space[para_name] = hp.choice(para_name, para_value)
                elif para_type == 'uniform':
                    low, high = para_value.strip().split(',')
                    space[para_name] = hp.uniform(para_name, float(low), float(high))
                elif para_type == 'quniform':
                    low, high, q = para_value.strip().split(',')
                    space[para_name] = hp.quniform(para_name, float(low), float(high), float(q))
                elif para_type == 'loguniform':
                    low, high = para_value.strip().split(',')
                    space[para_name] = hp.loguniform(para_name, float(low), float(high))
                else:
                    raise ValueError('Illegal param type [{}]'.format(para_type))
        return space

    @staticmethod
    def params2str(params):
        params_str = ''
        for param_name in params:
            params_str += param_name + ':' + str(params[param_name]) + ', '
        return params_str[:-2]

    @staticmethod
    def _print_result(result_dict: dict):
        logger = getLogger()
        logger.info('current best valid score: %.4f' % result_dict['best_valid_score'])
        logger.info('current test result:')
        logger.info(result_dict['test_result'])

    def export_result(self, output_file=None):
        with open(output_file, 'w') as fp:
            for params in self.params2result:
                fp.write(params + '\n')
                fp.write('Test result:\n' + str(self.params2result[params]['test_result']) + '\n')

    def fn(self, params):
        import hyperopt
        config_dict = params.copy()
        params_str = self.params2str(params)
        self._logger.info('running parameters:')
        self._logger.info(str(config_dict))
        result_dict = self.objective_function(
            task=self.task, model_name=self.model_name, dataset_name=self.dataset_name,
            config_file=self.config_file, saved_model=self.saved_model, train=self.train,
            other_args=self.other_args, hyper_config_dict=config_dict)
        self.params2result[params_str] = result_dict
        score = result_dict['best_valid_score']

        if not self.best_score:
            self.best_score = score
            self.best_params = params
            self._print_result(result_dict)
        elif score < self.best_score:
            self.best_score = score
            self.best_params = params
            self._print_result(result_dict)
        return {'loss': score, 'status': hyperopt.STATUS_OK}

    def run(self):
        from hyperopt import fmin
        fmin(self.fn, self.space, algo=self.algo, max_evals=self.max_evals)
