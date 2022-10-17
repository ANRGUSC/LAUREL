from functools import partial
from envs.multiagentenv import MultiAgentEnv
from envs.games.predator_prey_vec import PredatorPreyEnvVec
from envs.games.lumber_jack import LumberjacksEnv
from envs.wrapper import Wrapper
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    wifi_args = None
    if 'wifi_args' in kwargs:
        wifi_args = kwargs.pop('wifi_args')
    env_game = env(**kwargs)
    if wifi_args is not None:
        env_wrapper = Wrapper(env_game, wifi_args)
    else:
        env_wrapper = env_game
    return env_wrapper

REGISTRY = {
    "pp":   partial(env_fn, env=PredatorPreyEnvVec),
    "lj":   partial(env_fn, env=LumberjacksEnv)
}

if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH",
        os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )
