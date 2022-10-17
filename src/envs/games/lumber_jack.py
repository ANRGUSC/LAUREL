from collections import namedtuple
from envs.multiagentenv import MultiAgentEnv
import numpy as np
import torch
from envs.wireless.agents import Agents
from dataclasses import dataclass
from envs.games.utils import timeit
from typing import Dict, Set
from collections import defaultdict


@dataclass
class ACT:
    STAY  = 4
    TOP   = 0
    DOWN  = 2
    LEFT  = 3
    RIGHT = 1
    TREE_OBS_OFFSET = {0: 0, 1: 1, 2: 2, 3: 3}     
    OFF_X = {STAY:  0, TOP: -1, DOWN:  1, LEFT:  0, RIGHT:  0}      
    OFF_Y = {STAY:  0, TOP:  0, DOWN:  0, LEFT: -1, RIGHT:  1}

@dataclass
class REWARD:
    CUT_TREE_REWARD = 0.5
    POS_TREE_REWARD = 0.05
    TIMESTEP_PENALTY = -0.1


@dataclass
class BlockedActionInfo:
    off_x: int
    off_y: int
    blocked_action: int
    observed_tree_id: int


class LumberjacksEnv(MultiAgentEnv):
    def __init__(self, **config):
        self.__version__ = "0.0.1"
        self.name = "LJ-serial"
        self.serial_exec = True

        self.batch_size = 1 if 'para_vec' not in config.keys() else config['para_vec']
        assert self.batch_size == 1, "only support non-vectorized version"
        self.explicit_cut_tree_action = config["explicit_cut_tree_action"]

        self.vision = config['vision']
        self.dim = config['size']
        self.dims = (self.dim, self.dim)

        self.INVALID_TREE_POS = -2      

        self.BASE = np.prod(self.dims)
        self.OBSTACLE_CLASS = 0 + self.BASE
        self.OUTSIDE_CLASS = 1 + self.BASE
        self.TREE_CLASS_START = 2 + self.BASE
        num_tree_classes = 1 if self.vision > 0 else 4
        self.PREDATOR_CLASS = 1 + num_tree_classes + 1 + self.BASE
        
        self.tree_cut_threshold = 2

        
        self.npredator = config["nagents"]
        self.ntree = config["ntree"]

        self.obstacles = config['obstacles']        

        self.tree_attn = self.obstacles["attn_tree"]

        self.num_game_actions = 5

        if self.vision == 0:
            self.vocab_size = 1 + 4 + self.BASE + 1 + 1
        else:
            self.vocab_size = 1 + 1 + self.BASE + 1 + 1

        Agent = namedtuple("Agent", ["predator", "tree"])
        _ids_predator = np.arange(self.npredator)[np.newaxis, :]
        _ids_tree = np.arange(self.ntree)[np.newaxis, :]
        self.agent = Agent(
            predator=Agents(_ids_predator, None, can_comm=False),
            tree=Agents(_ids_tree, None, can_comm=False)
        )

        self.num_remain_trees = self.ntree
        self.episode_over = False
        self.episode_limit = config["limit"]
        self.episode_steps = 0     

        self.last_blocked_action = None    
        self.flag_observed_tree = np.zeros((self.npredator, self.ntree), dtype=np.bool)

    @staticmethod
    def init_args(parser):
        pass
        
    def reset(self):
        self.num_remain_trees = self.ntree
        self.episode_over = False
        self.episode_steps = 0

        num_samples = self.npredator + self.ntree
        pos_raw = np.random.choice(np.prod(self.dims), size=num_samples, replace=False)
        x, y = np.unravel_index(pos_raw, self.dims)
        pos_all = np.concatenate((x[..., np.newaxis], y[..., np.newaxis]), axis=-1)
        self.agent.predator.pos = pos_all[np.newaxis, :self.npredator, :]
        self.agent.tree.pos = pos_all[np.newaxis, self.npredator:, :]
        self.agent.predator.mask_freeze[:] = False     
        self.agent.tree.mask_freeze[:] = False
        self._set_grid()
        self.states = self.get_state()
        self.reward = None
        self.obs_pos = self.agent.tree.pos    
        self.last_blocked_action = None     
        self.flag_observed_tree = np.zeros((self.npredator, self.ntree), dtype=np.bool)
        return None, self.states

    @property
    def n_agents(self):
        return self.npredator

    @property
    def obs_start_pos(self):
        return self.obs_pos

    @property
    def obs_end_pos(self):
        return self.obs_pos
    
    @property
    def obs_attn(self):
        return [self.tree_attn] * self.agent.tree.pos.shape[1]
    
    @property
    def idx_batch_alive(self):
        if self.episode_over:
            return np.array([])
        else:
            return np.array([0])

    def get_agent_viz_state(self, mask):
        return {
            "predator": self.agent.predator.pos[mask].copy(), 
            "tree": self.agent.prey.pos[mask].copy()
        }
    
    def _set_grid(self):
        """
        Setting a static data struct -- independent of who is alive in a batch
        """
        grid = np.arange(self.BASE).reshape(self.dims)
        grid = np.pad(grid, self.vision, 'constant', constant_values=self.OUTSIDE_CLASS)
        _d1, _d2 = grid.shape
        self.empty_bool_base_grid = np.zeros((_d1, _d2, self.vocab_size), dtype=np.int)
        self.empty_bool_base_grid[
            np.arange(_d1)[:, np.newaxis], 
            np.arange(_d2)[np.  newaxis, :],
            grid       
        ] = 1
        self.empty_bool_base_grid = self.empty_bool_base_grid[np.newaxis, ...]

    def get_state(self):
        loc_map = self.onehot_pos(self.agent.predator.pos, flatten=True)
        glb_map = loc_map.sum(axis=1, keepdims=True)   
        tree_map = self.onehot_pos(self.agent.tree.pos, value_=-0.5, flatten=True)
        tree_map = tree_map.sum(axis=1, keepdims=True)
        glb_map += tree_map
        glb_map = np.broadcast_to(glb_map, (1, self.npredator, glb_map.shape[-1]))
        global_state = np.concatenate([loc_map, glb_map], axis=-1)
        return global_state

    def get_state_size(self):
        """
        assume the global state contains the full n x n map.
        """
        return (self.npredator, self.dim * self.dim * 2)
    
    def get_avail_actions(self):
        return [[1] * self.num_game_actions for _ in range(self.npredator)]
    
    def get_obs_size(self):
        return (2 * self.vision + 1)**2 * self.vocab_size
    
    @timeit
    def step(self, action):
        self.episode_steps += 1
        assert len(action.shape) == 2 and action.shape[0] == 1
        assert not self.episode_over, "Episode is done!"
        
        action = action % self.num_game_actions
        self._take_actions(action)  
        self._set_reward()

        if self.episode_steps == self.episode_limit:
            self.episode_over = True
        return self.reward, np.array([self.episode_over]), {}

    def _take_actions(self, action):
        assert action.shape[0] == 1
        action[np.where(self.agent.predator.mask_freeze)] = ACT.STAY

        act, loc = np.unique(action, return_inverse=True)
        off_x = np.array([ACT.OFF_X[a] for a in act])
        off_y = np.array([ACT.OFF_Y[a] for a in act])
        off_x = off_x[loc].reshape(action.shape)      
        off_y = off_y[loc].reshape(action.shape)      

        temp_pos_x = self.agent.predator.pos[..., 0] + off_x
        temp_pos_y = self.agent.predator.pos[..., 1] + off_y
        mask_invalid_move = (temp_pos_x[:, :, np.newaxis] == self.obs_pos[..., 0][:, np.newaxis,:]) \
                          * (temp_pos_y[:, :, np.newaxis] == self.obs_pos[..., 1][:, np.newaxis,:])
        blocked_ag_id, blocking_tree_id = np.where(mask_invalid_move[0])
        mask_invalid_move = mask_invalid_move.sum(axis=-1).astype(bool)
        self.last_blocked_action = {}
        for i, j in zip(blocked_ag_id, blocking_tree_id):
            self.last_blocked_action[i] = BlockedActionInfo(off_x[0, i], off_y[0, i], action[0, i], j)
        off_x = off_x * ~mask_invalid_move
        off_y = off_y * ~mask_invalid_move

        self.agent.predator.pos[..., 0] += off_x
        self.agent.predator.pos[..., 1] += off_y
        self.agent.predator.pos[..., 0] = np.clip(self.agent.predator.pos[..., 0], 0, self.dims[0] - 1)
        self.agent.predator.pos[..., 1] = np.clip(self.agent.predator.pos[..., 1], 0, self.dims[1] - 1)
    
    def get_obs(self):
        bool_base_grid = self.empty_bool_base_grid.copy()
        for ia in range(self.npredator):
            ipos = self.agent.predator.pos[0, ia]
            bool_base_grid[0, ipos[0] + self.vision, ipos[1] + self.vision, self.PREDATOR_CLASS] += 1
        for it in range(self.agent.tree.pos.shape[1]):
            ipos = self.agent.tree.pos[0, it]
            bool_base_grid[0, ipos[0] + self.vision, ipos[1] + self.vision, self.TREE_CLASS_START] += 1
        self.agent.predator.pos + np.arange(2 * self.vision + 1)
        obs_n_agents = []
        for ia, (ix, iy) in enumerate(self.agent.predator.pos[0]):
            obs_ag = bool_base_grid[
                0, 
                ix : ix + 2*self.vision + 1, 
                iy : iy + 2*self.vision + 1
            ].copy()
            if (
                self.vision == 0 
                and self.last_blocked_action is not None
                and ia in self.last_blocked_action
            ):
                blk_info = self.last_blocked_action[ia]
                if bool_base_grid[
                    0, 
                    ix + blk_info.off_x + self.vision, 
                    iy + blk_info.off_y + self.vision, 
                    self.TREE_CLASS_START
                ]:    
                    obs_ag[0, 0, self.TREE_CLASS_START] = 0
                    obs_ag[0, 0, self.TREE_CLASS_START + ACT.TREE_OBS_OFFSET[blk_info.blocked_action]] = 1
            obs_n_agents.append(obs_ag)
        return np.concatenate(obs_n_agents, axis=1)

    def _set_reward(self):
        reward = np.full(self.npredator, REWARD.TIMESTEP_PENALTY)
        tree_pos_prior = self.agent.tree.pos.copy()    
        if not self.explicit_cut_tree_action:
            for it, t_pos in enumerate(self.agent.tree.pos[0]):
                if t_pos[0] < 0 or t_pos[1] < 0:   
                    continue
                dist_manhattan = np.abs(self.agent.predator.pos[0] - t_pos[np.newaxis, :]).sum(axis=1)
                idx_arrive_ag = np.where(dist_manhattan <= 1)[0]
                if idx_arrive_ag.size >= self.tree_cut_threshold:
                    self.agent.tree.pos[0, it] = self.INVALID_TREE_POS
                    self.num_remain_trees -= 1
                    reward[idx_arrive_ag] += REWARD.CUT_TREE_REWARD
        else:
            tree_id_to_ag_cut_id = defaultdict(set)
            for a, blk_info in self.last_blocked_action.items():
                tree_id_to_ag_cut_id[blk_info.observed_tree_id].add(a)
            for i_tree, i_agcut in tree_id_to_ag_cut_id.items():
                if len(i_agcut) >= self.tree_cut_threshold:
                    self.agent.tree.pos[0, i_tree] = self.INVALID_TREE_POS
                    self.num_remain_trees -= 1
                    for ia in i_agcut:
                        reward[ia] += REWARD.CUT_TREE_REWARD
        if self.vision == 0:
            for a, blk_info in self.last_blocked_action.items():
                if not self.flag_observed_tree[a, blk_info.observed_tree_id]:
                    reward[a] += REWARD.POS_TREE_REWARD
                    self.flag_observed_tree[a, blk_info.observed_tree_id] = 1
        else:
            breakpoint()   
            at_tree = (self.agent.predator.pos[:, 0][:, np.newaxis] == self.agent.tree.pos[:, 0][np.newaxis, :])\
                  and (self.agent.predator.pos[:, 1][:, np.newaxis] == self.agent.tree.pos[:, 1][np.newaxis, :])
            idx_at_tree = np.where(at_tree.sum(axis=-1))[0]
            reward[idx_at_tree] += REWARD.POS_TREE_REWARD
        self.reward = reward[np.newaxis, ...].sum(axis=-1)
        if self.num_remain_trees <= 0:
            self.episode_over = True

    def _debug_tree_found_and_cut(self, tree_pos_prior, reward_all_ag):
        if self.reward.sum() > -0.5:
            print(f"reward is {reward_all_ag}, self.reward is {self.reward}, step is {self.episode_steps}")            
            if -0.05 in reward_all_ag:
                print(f"found\n{self.agent.predator.pos}\n{self.last_blocked_action}")
            if 0.4 in reward_all_ag:
                print(f"cut\n{self.agent.predator.pos}\n{tree_pos_prior}")
            breakpoint()    
        if self.episode_over or (self.episode_steps >= 38):
            print("episdoe over")

    def get_total_actions(self):
        return self.num_game_actions

    def onehot_pos(self, pos, value_=1, flatten=False):
        assert pos.shape[0] == 1, "we only support batch size of 1!"
        nagents = pos.shape[1]
        pos_map = np.zeros((1, nagents, self.dim, self.dim))
        pos_map[
            np.arange(1)[:, np.newaxis],
            np.arange(nagents)[np.newaxis, :],
            pos[..., 0],
            pos[..., 1]
        ] = value_
        if flatten:
            pos_map = pos_map.reshape(1, nagents, -1)
        return pos_map

    def close(self):
        pass