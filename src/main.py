import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

from run import run

SETTINGS['CAPTURE_MODE'] = "fd"
logger = get_logger()

ex = Experiment("LAUREL")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            if len(_v.split('=')) == 2:
                config_name = _v.split("=")[1]
            del params[_i]
            break
    if config_name is not None and subfolder is not None:
        filename = os.path.join(os.path.dirname(__file__), 'config', subfolder, f'{config_name}.yaml')
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict, filename
    else:
        return config_name


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == '__main__':
    params = deepcopy(sys.argv)

    config_dict, file_rl_config = _get_config(params, "--rl-config", ".")
    env_config, file_env_config = _get_config(params, "--env-config", "envs")
    alg_config, file_alg_config = _get_config(params, "--config", "algs")
    gpu_number = _get_config(params, "--cuda", None)
    checkpoint_path = _get_config(params, "--checkpoint_path", None)
    logger_repetition_key = _get_config(params, "--repetition_key", None)
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)
    config_dict['file_env_config'] = file_env_config
    config_dict['file_alg_config'] = file_alg_config
    config_dict['cuda'] = gpu_number
    config_dict['checkpoint_path'] = checkpoint_path
    if logger_repetition_key is not None:
        config_dict['repetition_key'] = logger_repetition_key
    if '--evaluate' in sys.argv:
        _get_config(params, '--evaluate', None)    
        config_dict['evaluate'] = True
    if '--viz' in sys.argv:
        _get_config(params, '--viz', None)
        config_dict['visualize'] = True

    assert (set(alg_config.keys()).issubset(set(config_dict.keys()))) and (set(env_config.keys()).issubset(set(config_dict.keys())))

    ex.add_config(config_dict)

    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)

