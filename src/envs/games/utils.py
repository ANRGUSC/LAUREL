from collections import defaultdict
from functools import wraps
from time import time


time_profile = {'env_game': defaultdict(float), 'env_wifi': defaultdict(float)}
map_cls2envtype = {
    'PredatorPreyEnvVec': 'env_game',
    'LumberjacksEnv': 'env_game',
    'EnvWirelessVector': 'env_wifi',
    'Agents': 'env_wifi',
    'pCSMAs': 'env_wifi',
    'slice_idx': 'env_wifi',
    'NetworkMeasure': 'env_wifi',
    'Buffer': 'env_wifi',
    'QueueHeader': 'env_wifi',
    'dBmSum_reduceat': 'env_wifi'
}

def timeit(func):
    """
    :param func: Decorated function
    :return: Execution time for the decorated function
    """
    @wraps(func)
    def wrapper_time(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        name_cls = func.__qualname__.split('.')[0]
        time_profile[map_cls2envtype[name_cls]][func.__qualname__] += end - start
        return result
    return wrapper_time