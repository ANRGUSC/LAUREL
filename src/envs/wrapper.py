import numpy as np
import torch
from inspect import getargspec
from envs.wireless.envs import EnvWirelessVector
from envs.wireless.agents import Agents
from envs.multiagentenv import MultiAgentEnv
from types import ModuleType
import math

import importlib

import envs.games.utils as utils

from collections import defaultdict



class Wrapper(MultiAgentEnv):
    '''
    for multi-agent
    '''
    def __init__(self, env, config_wifi, dtype_np=None, logger=None):
        self.msg_replay = config_wifi['msg_replay']
        self.env = env
        self.device = torch.device("cpu")
        self.comm_type = config_wifi['comm_type']
        assert self.comm_type in ['policy', 'always', 'never']
        if dtype_np is None:
            self.dtype_np = np.float32
        else:
            self.dtype_np = dtype_np
        self.logger = logger
        # --------------------
        self.with_comm = True      
        if config_wifi['name'].lower() != 'none' and self.comm_type != 'never':
            self.with_wireless_env = True
            if type(config_wifi['constant_file']) == str:
                self.wifi_const = importlib.import_module(f"envs.wireless.constants.{config_wifi['constant_file']}")
                self.wifi_const = vars(self.wifi_const)
            else:
                assert type(config_wifi['constant_file']) == dict
                self.wifi_const = config_wifi['constant_file']
        else:
            self.with_wireless_env = False
            self.wifi_const = None
        self.episode_limit = self.env.episode_limit
        self.n_agents = self.env.n_agents
        
        self.serial_exec = self.env.serial_exec
        self.env.serial_exec = False

    @property
    def num_actions(self):
        if hasattr(self.env.action_space, 'nvec'):
            return int(self.env.action_space.nvec[0])
        elif hasattr(self.env.action_space, 'n'):
            return self.env.action_space.n

    @property
    def dim_actions(self):
        if hasattr(self.env.action_space, 'nvec'):
            return self.env.action_space.shape[0]
        elif hasattr(self.env.action_space, 'n'):
            return 1

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):

        self.action_comm_trajectory = []
        obs = self.env.reset()
        self.predator_pos_prev = self.env.agent.predator.pos.copy()
        return obs, self.predator_pos_prev.astype(self.dtype_np)
    
    def get_obs(self):

        obs = self.env.get_obs()
        if self.serial_exec:
            obs = obs[0]
            obs = [obs[i] for i in range(self.n_agents)]
        return obs
    
    def get_obs_size(self):
        return self.env.get_obs_size()

    def get_state(self):
        global_state = self.env.get_state()
        if self.serial_exec:
            global_state = global_state[0]
        return global_state

    def get_state_size(self):
        return self.env.get_state_size()
    
    def get_avail_actions(self):
        game_action = self.env.get_avail_actions()
        assert type(game_action[0]) == list
        return [g * (1 + (self.comm_type == 'policy')) for g in game_action]     

    def get_total_actions(self):
        num_game_actions = self.env.get_total_actions()
        return num_game_actions * (1 + (self.comm_type == 'policy'))

    def record_timing(self):
        self.logger.record_timing(utils.time_profile)
        utils.time_profile = {'env_game': defaultdict(float), 'env_wifi': defaultdict(float)}

    def get_agent_viz_state(self, mask):
        return self.env.get_agent_viz_state(mask)    

    def step(self, action):

        if self.serial_exec:
            if type(action) == torch.Tensor:
                action = action.cpu().numpy()
            action = action[np.newaxis, ...]
        idx_batch_alive_b4_step = self.env.idx_batch_alive.copy()
        self.predator_pos_prev = self.env.agent.predator.pos[idx_batch_alive_b4_step].copy()
        if self.comm_type == 'policy':
            action_comm = action >= self.env.num_game_actions
        else:
            action_comm = (np.zeros(action.shape) + (self.comm_type == 'always')).astype(np.int)
        self.action_comm = action_comm
        self.action_comm_trajectory.append(action_comm.astype(np.float))
        batch_size, nagents =  self.env.agent.predator.pos[idx_batch_alive_b4_step].shape[:2]        
        if self.with_wireless_env:
            rss = np.full((batch_size, nagents, nagents), self.wifi_const['INVALID_RSS'])
            recv_mask = np.zeros((batch_size, nagents, nagents))
            grid_size = 100 / (self.env.dim - 1)

            id_agents = np.broadcast_to(np.arange(nagents)[np.newaxis, :], (batch_size, nagents))
            self.agents_wifi = Agents(id_agents, self.wifi_const, log_event=False) 
            self.env_wifi = EnvWirelessVector(agents_group={'comm': self.agents_wifi}, wifi_const=self.wifi_const) 
            self.agents_wifi.pos = self.predator_pos_prev * grid_size
            assert np.all(self.env.agent.predator.pos[idx_batch_alive_b4_step] * grid_size == self.env_wifi.agents['comm'].pos)
            if self.env.obs_pos is not None:
                obs_start_pos = self.env.obs_start_pos[idx_batch_alive_b4_step] * grid_size - grid_size/2
                obs_end_pos = self.env.obs_end_pos[idx_batch_alive_b4_step] * grid_size + grid_size/2
                obstacles = np.concatenate((obs_start_pos, obs_end_pos), axis=-1)
                obs_attn = self.env.obs_attn
                self.env_wifi.reset(obstacles, obs_attn)
            else:
                self.env_wifi.reset(np.array([])[np.newaxis, :], np.array([])[np.newaxis, :])
            if not ('SIMPLIFIED' in self.wifi_const.keys() and self.wifi_const['SIMPLIFIED']):
                ret_wifi = self.step_wifi(self.wifi_const, self.env_wifi, action_comm)
                ret_all_none = True
                if ret_wifi is not None:
                    rss[ret_wifi['indices']] = ret_wifi['values_rss']
                    recv_mask[ret_wifi['indices']] = ret_wifi['values_recv_mask']
                    ret_all_none = False
                if ret_all_none:
                    recv_mask = np.zeros((batch_size, nagents, nagents))
                    rss = np.full((batch_size, nagents, nagents), -np.inf)
            else:
                recv_mask, rss = self.env_wifi.step(num_updates=None, idx_send=None, pkt_send=None, action_comm=action_comm)
        else:
            rss = None
            recv_mask = np.repeat(action_comm[:, np.newaxis, :], nagents, axis=1)
            recv_mask[:, np.arange(nagents), np.arange(nagents)] = 0

        self.recv_mask = recv_mask if not self.serial_exec else [m for m in recv_mask[0]]
        msg_measure = np.zeros((batch_size,) + self.get_comm_measure_size())
        msg_measure[..., 0] = rss
        msg_measure[..., 1:3] = self.env.agent.predator.pos[:, np.newaxis, :, :]
        msg_measure[..., 3:5] = self.env.agent.predator.pos[:, :, np.newaxis, :]
        msg_measure = np.nan_to_num(msg_measure)
        msg_measure[..., 0] = (msg_measure[..., 0] + 100) / 120
        msg_measure[..., 1:5] = msg_measure[..., 1:5] / self.env.dim
        self.msg_measure = msg_measure if not self.serial_exec else [m for m in msg_measure[0]]

        reward, terminated, info = self.env.step(action)
        reward = reward if not self.serial_exec else reward[0]
        terminated = terminated if not self.serial_exec else terminated[0]
        return reward, terminated, info

    def step_wifi(self, wifi_const, env_wifi, action):
        idx_sender = np.where(action)
        id_sender = idx_sender[1]
        pkt_sender = {
            'id_sender'     : id_sender.flatten(),
            'id_data'       : np.zeros(len(id_sender), dtype=np.int),
            'id_receiver'   : np.full(len(id_sender), -1, dtype=np.int),                           
            'payload'       : np.ones(len(id_sender), dtype=np.int),      
            'mode'          : np.zeros(len(id_sender), dtype=np.int),    
            'length'        : np.full(len(id_sender), wifi_const['PKT_TIME']),
            'require_ack'   : np.ones(len(id_sender), dtype=np.bool),
            'rss'           : np.full(len(id_sender), -np.inf)
        }
        meta_csr_msg, pkt_csr_msg, meta_csr_ack, pkt_csr_ack = \
                env_wifi.step(num_updates=wifi_const['NUM_UPDATES'], idx_send=idx_sender, pkt_send=pkt_sender, action_comm=action)
        if meta_csr_msg['comm']['indptr'][-1] > 0:
            msg_cnt =  meta_csr_msg['comm']['indptr'][1:] -  meta_csr_msg['comm']['indptr'][:-1]
            msg_recv_ids = np.repeat(meta_csr_msg['comm']['idx_agent'], msg_cnt)
            msg_recv_batch = np.repeat(meta_csr_msg['comm']['idx_batch'], msg_cnt)
            msg_sender_ids = pkt_csr_msg['comm']['id_sender']
            msg_rss = pkt_csr_msg['comm']['rss']

            ret = {
                'indices': (msg_recv_batch, msg_recv_ids, msg_sender_ids),
                'values_recv_mask': 1,
                'values_rss': msg_rss
            }
        else:
            ret = None
        return ret

    def reward_terminal(self):
        if hasattr(self.env, 'reward_terminal'):
            return self.env.reward_terminal()
        else:
            return np.zeros(1)

    def _flatten_obs(self, obs):
        if isinstance(obs, tuple):
            raise NotImplementedError      
            _obs=[]
            for agent in obs:
                ag_obs = []
                for obs_kind in agent:
                    ag_obs.append(np.array(obs_kind).flatten())
                _obs.append(np.concatenate(ag_obs))
            obs = np.stack(_obs)
        obs = obs.reshape(obs.shape[0], obs.shape[1], self.observation_dim)        
        obs = torch.from_numpy(obs.astype(self.dtype_np)).to(self.device)
        return obs

    def get_stat(self):
        if hasattr(self.env, 'stat'):
            self.env.stat.pop('steps_taken', None)
            return self.env.stat
        else:
            return dict()

    def get_comm_measure_size(self):
        return (self.env.n_agents, self.env.n_agents, 5)

    def get_comm_msg_augment_size(self):
        if self.env.name == 'PP' or self.env.name.startswith('LJ'):
            return self.env.dim**2
        else:
            raise NotImplementedError

    def augment_comm_msg(self, comm_msg):
        assert len(comm_msg.shape) == 2 and comm_msg.shape[0] == self.env.n_agents
        if self.env.name == 'PP' or self.env.name.startswith("LJ"):
            pos_1hot = self.env.onehot_pos(self.env.agent.predator.pos, flatten=True)[0]
            comm_msg = np.concatenate([pos_1hot, comm_msg], axis=1)
        else:
            raise NotImplementedError
        return [m for m in comm_msg]

    def avg_comm_action(self):
        if len(self.action_comm_trajectory) > 0:
            if not self.serial_exec:
                return None
            assert self.action_comm_trajectory[0].size == self.env.n_agents
            return np.sum(self.action_comm_trajectory) / len(self.action_comm_trajectory) / self.env.n_agents
        else:
            return None
                
    def close(self):
        pass