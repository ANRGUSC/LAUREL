from envs.multiagentenv import MultiAgentEnv

import random
import math
import curses
from collections import namedtuple
import torch

import numpy as np
from envs.wireless.agents import Agents
from envs.wireless.envs import EnvWirelessVector
from dataclasses import dataclass
from envs.games.utils import timeit

from envs.games.predator_prey_obs import (
    generate_all_obstacles,
    generate_random_adj_prey, 
    generate_random_predator_prey
)

@dataclass
class ACT:
    STAY  = 4
    TOP   = 0
    DOWN  = 2
    LEFT  = 3
    RIGHT = 1
    OFF_X = {STAY:  0, TOP: -1, DOWN:  1, LEFT:  0, RIGHT:  0}      
    OFF_Y = {STAY:  0, TOP:  0, DOWN:  0, LEFT: -1, RIGHT:  1}


class PredatorPreyEnvVec(MultiAgentEnv):    

    def __init__(self, **config):
        self.__version__ = "0.0.1"
        self.batch_size = 1 if 'para_vec' not in config.keys() else config['para_vec']
        
        self.OBSTACLE_CLASS = 0
        self.OUTSIDE_CLASS = 1
        self.PREY_CLASS = 2
        self.PREDATOR_CLASS = 3
        self.TIMESTEP_PENALTY = -0.1
        self.PREY_REWARD = 0
        self.SUCCESS_REWARD = 1
        self.POS_PREY_REWARD = 0.05
        self.episode_over = np.zeros(self.batch_size, dtype=np.bool)

        self.dim = config['size']
        self.vision = config['vision']
        self.moving_prey = config['moving_prey']

        self.nprey = config['nenemies']
        self.n_agents = self.npredator = config['nagents']
        self.dims = dims = (self.dim, self.dim)
        self.stay = not config['no_stay']
        self.obstacles = config['obstacles']
        if self.obstacles == {}:
            self.obstacles = None

        assert not config['moving_prey']

        self.num_game_actions = 5      
        self.BASE = (dims[0] * dims[1])
        self.OUTSIDE_CLASS += self.BASE
        self.PREY_CLASS += self.BASE
        self.PREDATOR_CLASS += self.BASE
        self.OBSTACLE_CLASS += self.BASE

        self.vocab_size = 1 + 1 + self.BASE + 1 + 1
 


        Agent = namedtuple("Agent", ['predator', 'prey'])


        _ids_predator = np.broadcast_to(np.arange(self.npredator)[np.newaxis, :], (self.batch_size, self.npredator))
        _ids_prey = np.broadcast_to(self.npredator + np.arange(self.nprey)[np.newaxis, :], (self.batch_size, self.nprey))
        self.agent = Agent(
                predator=Agents(_ids_predator, None, can_comm=False), 
                prey=Agents(_ids_prey, None, can_comm=False)       
            )
        
 


        self.episode_over = np.zeros(self.batch_size, dtype=np.bool)
        self.name = f"PP" 

        self.FREEZE_UPON_CAUGHT = True 

        self.episode_limit = config['limit']
        self._episode_steps = np.zeros(self.batch_size, dtype=np.int32)
        self.serial_exec = config['serial_exec']
        self.idx_batch_alive = np.arange(self.batch_size)
        
    @staticmethod
    def init_args(parser):
        pass
        
    def get_agent_viz_state(self, mask):
        return {"predator": self.agent.predator.pos[mask].copy(), 
                "prey": self.agent.prey.pos[mask].copy()}
    
    @timeit
    def step(self, action):
      
        self._episode_steps[self.idx_batch_alive] += 1
        if self.serial_exec:
            if type(action) == torch.Tensor:
                action = action.cpu().numpy()
            action = action[np.newaxis, ...]

        assert len(action.shape) == 2
        if np.all(self.episode_over):
            raise RuntimeError("Episode is done")

        action = action % self.num_game_actions    
        
        assert not self.moving_prey, "Moving prey not implemented"
        self._take_actions(self.agent.predator, action)

 
        
        idx_still_alive = self._set_reward()
   
        _reward = self.reward if not self.serial_exec else self.reward[0]
        self.episode_over[self._episode_steps == self.episode_limit] = True
        _terminated = self.episode_over if not self.serial_exec else self.episode_over[0]
        return _reward, _terminated, {}
    
    def reset(self):
      
        self.episode_over = np.zeros(self.batch_size, dtype=np.bool)
        self.idx_batch_alive = np.arange(self.batch_size)
        self.batch_size_alive = self.batch_size
    
        self._episode_steps = np.zeros(self.batch_size, dtype=np.int32)

        if self.obstacles is None:
            obs_pos = None
        else:
            obs_pos, mask_batch_obs_horizon = generate_all_obstacles(self.obstacles, self.batch_size, self.dim)
        self.obs_pos = obs_pos        
        if obs_pos is not None:
          
            self.obs_start_pos = self.obs_pos[:, :, 0, :]
            self.obs_end_pos = self.obs_pos[:, :, -1, :]
            self.obs_attn = self.obstacles['attn']
            self.mask_batch_obs_horizon = mask_batch_obs_horizon
            
            if self.obstacles['random'] == "adjacent":
                prey_pos, idx_prey_invalid = generate_random_adj_prey(self.obs_pos, self.mask_batch_obs_horizon)
                self.agent.prey.pos = prey_pos[:, np.newaxis, :].astype(int)
            else:
                idx_prey_invalid = None            
     
            self.obs_all_pos = self.obs_pos.reshape(self.batch_size, -1, 2)
            generate_random_predator_prey(
                self.obs_all_pos, self.batch_size, self.dim, 
                self.agent.predator.pos, self.agent.prey.pos, idx_prey_invalid
            )
            self.agent.predator.pos = self.agent.predator.pos.astype(int)
            self.agent.prey.pos = self.agent.prey.pos.astype(int)
        else:
            num_samples = self.npredator + self.nprey
            num_choices = np.prod(self.dims)
            pos_raw = np.repeat(np.arange(num_choices)[np.newaxis, :], self.batch_size, axis=0)
            rng = np.random.default_rng()
            pos_raw = rng.permuted(pos_raw, axis=1)
            pos_raw = pos_raw[:, :num_samples]

            x, y = np.unravel_index(pos_raw, self.dims)
            pos_all = np.concatenate((x[:, :, np.newaxis], y[:, :, np.newaxis]), axis=-1)

            self.agent.predator.pos = pos_all[:, :self.npredator, :]
            self.agent.prey.pos = pos_all[:, self.npredator:, :]

        self.agent.predator.mask_freeze[:] = False
        self.agent.prey.mask_freeze[:] = False
        self._set_grid()
        self.obs = self.get_obs()
        self.states = self.get_state()
        self.reward = None
        return self.obs, self.states
        

    def seed(self):
        pass

    def _set_grid(self):

        grid = np.arange(self.BASE).reshape(self.dims)

        grid = np.pad(grid, self.vision, 'constant', constant_values = self.OUTSIDE_CLASS)
        _d1, _d2 = grid.shape
        self.empty_bool_base_grid = np.zeros((_d1, _d2, self.vocab_size), dtype=np.int)
        self.empty_bool_base_grid[
                np.arange(_d1)[:, np.newaxis], 
                np.arange(_d2)[np.newaxis, :],
                grid       
            ] = 1
        self.empty_bool_base_grid = np.repeat(self.empty_bool_base_grid[np.newaxis, :, :, :], self.batch_size, axis=0)   

        if self.obs_pos is not None:
            idx_obs_dim0 = np.repeat(np.arange(self.batch_size)[:, np.newaxis], np.prod(self.obs_pos.shape[1:3]), axis=-1)
            idx_obs_dim1 = self.vision + self.obs_pos[:, :, :, 0].reshape(self.batch_size, -1)
            idx_obs_dim2 = self.vision + self.obs_pos[:, :, :, 1].reshape(self.batch_size, -1)

            self.empty_bool_base_grid[idx_obs_dim0, idx_obs_dim1.astype(int), idx_obs_dim2.astype(int), self.OBSTACLE_CLASS] = 1 


    def get_obs(self):     



        if self.idx_batch_alive.size > 0:
            _idx_batch = self.idx_batch_alive
            _batch_size = self.batch_size_alive
        else:
            _idx_batch = np.arange(self.batch_size)   
            _batch_size = self.batch_size
        bool_base_grid = np.empty_like(self.empty_bool_base_grid[_idx_batch])
        bool_base_grid[:] = self.empty_bool_base_grid[_idx_batch]
        _idx0 = np.broadcast_to(np.arange(_batch_size)[:, np.newaxis], (_batch_size, self.npredator))
        _idx1 = self.agent.predator.pos[_idx_batch, :, 0] + self.vision        
        _idx2 = self.agent.predator.pos[_idx_batch, :, 1] + self.vision       
        _idx3 = self.PREDATOR_CLASS
        _idx_unique, _cnt_unique = np.unique(np.stack([_idx0.flatten(), _idx1.flatten(), _idx2.flatten()]), axis=1, return_counts=True)
        bool_base_grid[_idx_unique[0], _idx_unique[1], _idx_unique[2], _idx3] += _cnt_unique                            
        _idx0 = np.broadcast_to(np.arange(_batch_size)[:, np.newaxis], (_batch_size, self.nprey))
        _idx1 = self.agent.prey.pos[_idx_batch, :, 0] + self.vision     
        _idx2 = self.agent.prey.pos[_idx_batch, :, 1] + self.vision
        _idx3 = self.PREY_CLASS
        _idx_unique, _cnt_unique = np.unique(np.stack([_idx0.flatten(), _idx1.flatten(), _idx2.flatten()]), axis=1, return_counts=True)
        bool_base_grid[_idx_unique[0], _idx_unique[1], _idx_unique[2], _idx3] += _cnt_unique
        offset = np.arange(2 * self.vision + 1)[np.newaxis, np.newaxis, :]
        idx_obs_x = (self.agent.predator.pos[_idx_batch, :, 0][:, :, np.newaxis] + offset)[:, :, :, np.newaxis]       
        idx_obs_y = (self.agent.predator.pos[_idx_batch, :, 1][:, :, np.newaxis] + offset)[:, :, np.newaxis, :]       
        idx_batch = np.arange(_batch_size)[:, np.newaxis, np.newaxis, np.newaxis]
        if self.obs_pos is None:
            assert (np.where(self.empty_bool_base_grid[0,:,:,-4])[0]).size == 0, "this dim is not used when no obstacles"
        obs = bool_base_grid[idx_batch, idx_obs_x, idx_obs_y]
        b, n = obs.shape[:2]
        obs = obs.reshape(b, n, -1)
        if self.serial_exec:
            obs = obs[0]
            obs = [obs[i] for i in range(self.n_agents)]
        return obs

    def get_obs_size(self):
        return (2 * self.vision + 1)**2 * self.vocab_size

    def _take_actions(self, agent_group, acts):
 
        assert acts.shape[0] == self.batch_size_alive
        acts[np.where(agent_group.mask_freeze[self.idx_batch_alive])] = ACT.STAY
        
        act, loc = np.unique(acts, return_inverse=True)
        off_x = np.array([ACT.OFF_X[a] for a in act])
        off_y = np.array([ACT.OFF_Y[a] for a in act])
        off_x = off_x[loc].reshape(acts.shape)    
        off_y = off_y[loc].reshape(acts.shape)     
        
        if self.obs_pos is not None:
            temp_pos_x = agent_group.pos[self.idx_batch_alive, :, 0] + off_x
            temp_pos_y = agent_group.pos[self.idx_batch_alive, :, 1] + off_y
            mask_invalid_move = (temp_pos_x[:, :, np.newaxis] == self.obs_all_pos[self.idx_batch_alive, :, 0][:, np.newaxis,:]) * (temp_pos_y[:, :, np.newaxis] == self.obs_all_pos[self.idx_batch_alive, :, 1][:, np.newaxis,:])
            mask_invalid_move = mask_invalid_move.sum(axis=-1).astype(bool)
            off_x = off_x * ~mask_invalid_move
            off_y = off_y * ~mask_invalid_move

        agent_group.pos[self.idx_batch_alive, :, 0] += off_x
        agent_group.pos[self.idx_batch_alive, :, 1] += off_y
        agent_group.pos[self.idx_batch_alive, :, 0] = np.clip(agent_group.pos[self.idx_batch_alive, :, 0], 0, self.dims[0] - 1)
        agent_group.pos[self.idx_batch_alive, :, 1] = np.clip(agent_group.pos[self.idx_batch_alive, :, 1], 0, self.dims[1] - 1)
        


    def _set_reward(self):
      
        reward = np.full((self.batch_size_alive, self.npredator), self.TIMESTEP_PENALTY)
        pos_predator = self.agent.predator.pos[self.idx_batch_alive, :, np.newaxis, :]  
        pos_prey = self.agent.prey.pos[self.idx_batch_alive, np.newaxis, :, :]         
        match_xy = (pos_predator == pos_prey).astype(np.bool)       
        match_xy = match_xy[:, :, :, 0] * match_xy[:, :, :, 1]     
        match_per_predator = match_xy.sum(axis=-1)                  
        idx_success_predator = np.where(match_per_predator)        
        num_predator_on_prey = match_per_predator.sum(axis=-1)     
        num_success_predator_on_prey = num_predator_on_prey[np.where(num_predator_on_prey > 0)]

        reward[idx_success_predator] -= self.TIMESTEP_PENALTY      
        idx0 = self.idx_batch_alive[idx_success_predator[0]]
        idx1 = idx_success_predator[1]
        if self.FREEZE_UPON_CAUGHT:
            self.agent.predator.mask_freeze[idx0, idx1] = True    
        idx_new_over = self.idx_batch_alive[np.where(num_predator_on_prey == self.npredator)]
        idx_still_alive = np.where(num_predator_on_prey != self.npredator)[0]
        self.episode_over[idx_new_over] = True

        self.idx_batch_alive = np.where(~self.episode_over)[0]
        self.batch_size_alive = self.idx_batch_alive.size
        reward[idx_new_over] = self.SUCCESS_REWARD         
        self.reward = reward.sum(axis=-1)
        
        return idx_still_alive
    
    def onehot_pos(self, agent, value_=1, flatten=False):
        nagents = agent.pos.shape[1]
        pos_map = np.zeros((self.batch_size, nagents, self.dim, self.dim))
        pos_map[np.arange(self.batch_size)[:, np.newaxis],
                np.arange(nagents)[np.newaxis, :],
                agent.pos[..., 0],
                agent.pos[..., 1]] = value_
        if flatten:
            pos_map = pos_map.reshape(self.batch_size, nagents, -1)
        return pos_map
        


    def get_state(self):

        loc_map = self.onehot_pos(self.agent.predator, flatten=True)
        glb_map = loc_map.sum(axis=1, keepdims=True)
        glb_map += self.onehot_pos(self.agent.prey, value_=-0.5, flatten=True)
        # loc_map = self.onehot_pos(self.agent.predator, flatten=True)
        glb_map = np.broadcast_to(glb_map, (self.batch_size, self.n_agents, glb_map.shape[-1]))
        global_state = np.concatenate([loc_map, glb_map], axis=-1)
        return global_state


    def get_state_size(self):
        """
        assume the global state contains the full n x n map.
        """
        return (self.n_agents, self.dim * self.dim * 2)

    
    def get_avail_actions(self):
        return [[1] * self.num_game_actions for _ in range(self.n_agents)]
    
    def get_total_actions(self):
        return self.num_game_actions

    def close(self):
        pass

    