import datetime
import os
import sys
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.logging_base import LoggerBase
from utils.timehelper import time_left, time_str
from utils.rl_utils import get_gitrev_timestamp, setup_global_config
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

def run(_run, _config, _log):

    LoggerCuz, meta_config = setup_global_config(debug=('--debug' in sys.argv))
    if 'repetition_key' in _config:
        meta_config['logging']['logger_configs']['repetition_key'] = _config.pop('repetition_key')

    _config['dim_wifi_measure'] = 0
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    if args.use_cuda and args.cuda is not None and int(args.cuda) >= 0:
        args.device = f'cuda:{args.cuda}'
    else:
        args.device = 'cpu'

    # setup loggers
    gitrev, timestamp = get_gitrev_timestamp()
    config_env = {'game': args.env_args,
                  'wifi': {'name': 'none'} if not hasattr(args, 'wifi_args') else args.wifi_args}
    if 'name' not in config_env['game']:
        config_env['game']['name'] = args.env
    logger_customize = LoggerCuz(
        {'logger_key'   : '',
         'args'         : args,
         'env'          : config_env,
         'arch'         : {},
         'params'       : {}},
        meta_config['logging']['dir']['local'], None, timestamp, gitrev,
        **meta_config['logging']['logger_configs']
    )
    logger_customize.cpy_cfg_yml(_config['file_alg_config'])
    logger_customize.cpy_cfg_yml(_config['file_env_config'])
    logger_customize.cpy_cfg_yml(_config['file_rl_config'])
    # logger_customize.cpy_cfg_yml(f'./src/config/default.yaml')
    logger = Logger(_log, logger_customize=logger_customize)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")
    if logger.logger_customize is not None:
        logger.logger_customize.end_training('finished')

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner, logger=None, render=False):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True, render=render)
        runner.t_env += 1
    if args.save_replay:
        runner.save_replay()
    
    if logger is not None:
        logger.log_stat("episode", 1, 0)
        logger.print_recent_stats()
    runner.close_env()

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    if hasattr(runner.env, 'with_comm'):
        scheme["comm_mask"] = {'vshape': (args.n_agents, args.n_agents), 'dtype': th.bool}
        scheme["comm_measure"] = {'vshape': runner.env.get_comm_measure_size(), 'dtype': th.float32}
        args.dim_wifi_measure = scheme['comm_measure']['vshape'][-1]
        if runner.env.msg_replay == 'stored':
            scheme["comm_msg"] = {
                'vshape': (
                    args.n_agents, args.rnn_hidden_dim + runner.env.get_comm_msg_augment_size()
                ), 
                'dtype': th.float32
            }
            if args.agent == 'tar-rnn':
                scheme["attn_msg"] = {
                    "vshape": (args.n_agents, args.rnn_hidden_dim), "dtype": th.float32
                }
            args.dim_msg = scheme['comm_msg']['vshape'][-1]
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(
        scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device
    )

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    ## visualizer
    if hasattr(args, 'visualize') and args.visualize:
        from envs.visualizer.curses import VisualizerGridCurses as Viz
        style_agent = {'predator'   : {'color': 'cyan', 'marker': 'X'}, 
                       'prey'       : {'color': 'red',  'marker': '+'},
                       'obstacle'   : {'color': 'yellow', 'marker': '#'},
                       'wanna_send' : {'color': 'white', 'marker': 'X'}}
        viz = Viz(runner.env.env, *runner.env.env.dims, style_agent)
    else:
        args.visualize = False
        from envs.visualizer.base import VisualizerGridNull as Viz
        viz = Viz(runner.env.env, 0, 0, {})
    runner.viz = viz

    if args.checkpoint_path is not None and args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))
        if len(timesteps) > 0:
            if args.load_step == 0:
                timestep_to_load = max(timesteps)
            else:
                timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

            model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))
        else:
            model_path = args.checkpoint_path
        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            import numpy as np
            evaluate_sequential(args, runner, logger=logger, render=args.visualize)
            ep_len = [len(a) for a in runner.action_per_step]
            all_comm_actions = np.zeros((len(runner.action_per_step), max(ep_len), runner.action_per_step[0][0].size))
            for i, a in enumerate(runner.action_per_step):
                for j, aa in enumerate(a):
                    all_comm_actions[i, j] = aa
            step_first_catch_prey = np.array(runner.step_first_catch_prey)
            step_first_catch_prey[step_first_catch_prey <= -1] = max(ep_len)
            len_stat = np.bincount(step_first_catch_prey)        # ep_len
            nagents = all_comm_actions.shape[-1]
            str_csv = ['ep,' + ','.join(f'agent_avg_{i},agent_std_{i}' for i in range(nagents))+',dummy']
            data = []
            for i in range(nagents):
                ag_comm_action = all_comm_actions[..., i]
                ag_comm_avg, ag_comm_std = ag_comm_action.mean(axis=0), ag_comm_action.std(axis=0)
                data += [ag_comm_avg, ag_comm_std]
            data = [np.arange(data[0].size)] + data + [np.zeros(data[0].size)]
            ag_comm_data = np.vstack(data).T
            for dl in ag_comm_data:
                str_csv.append(','.join(str(dd) for dd in dl))
            with open('plot_when_breakdown.csv', 'w') as fout:
                fout.write('\n'.join(str_csv))
            str_csv = ['ep,term_cnt']
            for i,s in enumerate(len_stat):
                str_csv.append(f'{i},{s/sum(len_stat)}')
            with open('plot_when_breakdown_eplen.csv', 'w') as fout:
                fout.write('\n'.join(str_csv))
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info(f"Beginning training for {args.t_max} timesteps")

    while runner.t_env <= args.t_max:

        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        buf_can_sample = buffer.can_sample(args.batch_size)
        if buf_can_sample:
            episode_sample = buffer.sample(args.batch_size)

            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)
            learner.train(episode_sample, runner.t_env, episode)

        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T >= args.test_interval) and buf_can_sample:
            assert learner.log_stats_t == runner.t_env
            logger.console_logger.info(f"t_env: {runner.t_env} / {args.t_max}")
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)
            logger.update_test_epoch()
            logger.logger_customize.update_best_model(
                    logger.logger_customize.epoch_current, learner)
            
        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
