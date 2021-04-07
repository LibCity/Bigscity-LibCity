import numpy as np
from functools import partial
from logging import getLogger
import hyperopt
from hyperopt import hp, fmin, tpe, atpe, rand
from hyperopt.pyll.base import Apply


def _recursivefindnodes(root, node_type='switch'):
    nodes = []
    if isinstance(root, (list, tuple)):
        for node in root:
            nodes.extend(_recursivefindnodes(node, node_type))
    elif isinstance(root, dict):
        for node in root.values():
            nodes.extend(_recursivefindnodes(node, node_type))
    elif isinstance(root, Apply):
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
    for node in _recursivefindnodes(space, 'switch'):
        # Find the name of this parameter
        paramnode = node.pos_args[0]
        assert paramnode.name == 'hyperopt_param'
        paramname = paramnode.pos_args[0].obj
        # Find all possible choices for this parameter
        values = [literal.obj for literal in node.pos_args[1:]]
        parameters[paramname] = np.array(range(len(values)))
    return parameters


def _spacesize(space):
    # Compute the number of possible combinations
    params = _parameters(space)
    return np.prod([len(values) for values in params.values()])


class ExhaustiveSearchError(Exception):
    pass


def _validate_space_exhaustive_search(space):
    from hyperopt.pyll.base import dfs, as_apply
    from hyperopt.pyll.stochastic import implicit_stochastic_symbols
    supported_stochastic_symbols = ['randint', 'quniform', 'qloguniform',
                                    'qnormal', 'qlognormal', 'categorical']
    for node in dfs(as_apply(space)):
        if node.name in implicit_stochastic_symbols:
            if node.name not in supported_stochastic_symbols:
                raise ExhaustiveSearchError('Exhaustive search is only possible \
                                            with the following stochastic symbols: '
                                            + ', '.join(supported_stochastic_symbols))


def exhaustive_search(new_ids, domain, trials, seed, nb_max_sucessive_failures=1000):
    from hyperopt import pyll
    from hyperopt.base import miscs_update_idxs_vals
    # Build a hash set for previous trials
    hashset = set([hash(frozenset([(key, value[0]) if len(value) > 0 else ((key, None))
                                   for key, value in trial['misc']['vals'].items()])) for trial in trials.trials])

    rng = np.random.RandomState(seed)
    rval = []
    for _, new_id in enumerate(new_ids):
        new_sample = False
        nb_sucessive_failures = 0
        while not new_sample:
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
                new_sample = True
            else:
                # Duplicated sample, ignore
                nb_sucessive_failures += 1

            if nb_sucessive_failures > nb_max_sucessive_failures:
                # No more samples to produce
                return []

        rval.extend(trials.new_trial_docs([new_id], [None], [new_result], [new_misc]))
    return rval


class HyperTuning:
    """
    自动调参

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
        self.params2result = {}  # 每一种参数组合对应的最小验证集误差等结果

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
                self.algo = partial(exhaustive_search, nb_max_sucessive_failures=1000)
                self.max_evals = _spacesize(self.space)
            elif algo == 'tpe':
                self.algo = tpe.suggest
            elif algo == 'atpe':
                self.algo = atpe.suggest
            elif algo == 'random_search':
                self.algo = rand.suggest
            else:
                raise ValueError('Illegal hyper algorithm type [{}]'.format(algo))
        else:
            self.algo = algo

    @staticmethod
    def _build_space_from_file(file):
        space = {}
        with open(file, 'r') as fp:
            for line in fp:
                para_list = line.strip().split(' ')
                if len(para_list) < 3:
                    continue
                para_name, para_type, para_value = para_list[0], para_list[1], "".join(para_list[2:])
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
                    raise ValueError('Illegal parameter type [{}]'.format(para_type))
        return space

    @staticmethod
    def params2str(params):
        # dict to str
        params_str = ''
        for param_name in params:
            params_str += param_name + ':' + str(params[param_name]) + ', '
        return params_str[:-2]

    def save_result(self, filename=None):
        with open(filename, 'w') as fp:
            fp.write('best params: ' + str(self.best_params) + '\n')
            fp.write('best_valid_score: \n')
            fp.write(str(self.params2result[self.params2str(self.best_params)]['best_valid_score']) + '\n')
            fp.write('best_test_result: \n')
            fp.write(str(self.params2result[self.params2str(self.best_params)]['test_result']) + '\n')
            fp.write('----------------------------------------------------------------------------\n')
            fp.write('All parameters tune and result: \n')
            for params in self.params2result:
                fp.write(params + '\n')
                fp.write('Test result:\n' + str(self.params2result[params]['test_result']) + '\n')
        self._logger.info('hyper-tuning result is saved at {}'.format(filename))

    def fn(self, params):
        hyper_config_dict = params.copy()
        params_str = self.params2str(params)
        self._logger.info('running parameters:')
        self._logger.info(str(hyper_config_dict))
        result_dict = self.objective_function(
            task=self.task, model_name=self.model_name, dataset_name=self.dataset_name,
            config_file=self.config_file, saved_model=self.saved_model, train=self.train,
            other_args=self.other_args, hyper_config_dict=hyper_config_dict)
        self.params2result[params_str] = result_dict

        score = result_dict['best_valid_score']
        if not self.best_score:
            self.best_score = score
            self.best_params = params
        elif score < self.best_score:
            self.best_score = score
            self.best_params = params
        self._logger.info('current parameters:')
        self._logger.info(str(hyper_config_dict))
        self._logger.info('current best valid score: %.4f' % result_dict['best_valid_score'])
        self._logger.info('current test result:')
        self._logger.info(result_dict['test_result'])
        return {'loss': score, 'status': hyperopt.STATUS_OK}

    def start(self):
        fmin(self.fn, self.space, algo=self.algo, max_evals=self.max_evals)
