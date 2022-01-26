#from wireless.constants import *
import envs.wireless.utils as utils
import numpy as np
from dataclasses import dataclass, field, fields, InitVar
from envs.wireless.agents import Agents
from envs.wireless.data_struct import NetworkMeasure, QueueHeader
import envs.wireless.utils as utils
from typing import List, Union, Tuple
from collections import defaultdict
import time
from envs.games.utils import timeit

import copy
# np.random.seed(0)

@dataclass
class _Signal:     
  
    # data fields
    time_remaining                  : np.ndarray = np.array([])     
    id_data                         : np.ndarray = np.array([])     
    id_receiver                     : np.ndarray = np.array([])     
    payload                         : np.ndarray = np.array([])
    mode                            : np.ndarray = np.array([])
    length                          : np.ndarray = np.array([])
    require_ack                     : np.ndarray = np.array([])
    rss                             : np.ndarray = np.array([])
    # book-keeper
    mask_sender_step0               : np.ndarray = np.array([])     
    mask_active_sender              : np.ndarray = np.array([])     
    len_active_sender               : np.ndarray = np.array([])     
    _len_active_sender_cumsum       : np.ndarray = np.array([])     
    idx_terminating_sender          : Tuple[np.ndarray] = field(default_factory=tuple)      
    # constants
    T_TRANS                         : int = 1                       
    BANDWIDTH                       : int = 1
    INV_ID_RECV   = -1
    INV_ID_DATA   = -1
    INV_PAYLOAD   = -1
    INV_MODE      = -1
    INV_LENGTH    = -1
    # init fields
    _dim1                           : InitVar[int] = 0
    _dim2                           : InitVar[int] = 0
    def __post_init__(self, _dim1, _dim2):

        if _dim1 > 0 and _dim2 > 0:
            self.time_remaining = np.full((_dim1, _dim2), self.T_TRANS, dtype=np.int64)
            self.id_receiver = np.zeros((_dim1, _dim2), dtype=np.int64)
            self.id_data = np.zeros((_dim1, _dim2), dtype=np.int64)
            self.payload = np.zeros((_dim1, _dim2), dtype=np.int32)
            self.mode = np.zeros((_dim1, _dim2), dtype=np.int32)
            self.length = np.zeros((_dim1, _dim2))
            self.require_ack = np.zeros((_dim1, _dim2), dtype=np.bool)
            self.rss = np.zeros((_dim1, _dim2))
            self.mask_sender_step0 = np.zeros((_dim1, _dim2), dtype=np.bool)
            self.mask_active_sender = np.zeros((_dim1, _dim2), dtype=np.bool)
            self.len_active_sender = np.array([]) 
            self._len_active_sender_cumsum = np.array([-1], dtype=np.int64) 

    def set_len_cumsum(self):

        len_active_sender = self.mask_active_sender.sum(axis=1)
        self.len_active_sender = len_active_sender[len_active_sender > 0]
        self._len_active_sender_cumsum = np.roll(self.len_active_sender.cumsum(), 1)
        if len(self._len_active_sender_cumsum):  
            self._len_active_sender_cumsum[0] = 0
        else:
            self._len_active_sender_cumsum = np.array([-1])
    
    def add(self, pkts : dict, idxs : Union[tuple, list], t):

        assert np.all(self.mask_active_sender[idxs] == False)
        if pkts is not None:
            for k, v in pkts.items():
                if k != 'id_sender':
                    getattr(self, k)[idxs] = v
        self.time_remaining[idxs] = np.ceil(self.length[idxs] / self.BANDWIDTH).astype(self.time_remaining.dtype)
        self.mask_active_sender[idxs] = True
        self.set_len_cumsum()           

    def reset(self, hard=False):
        if not hard:
            assert self.idx_terminating_sender is not None
            if self.idx_terminating_sender[0].size == 0:
                self.idx_terminating_sender = None     
                return
            self.mask_active_sender[self.idx_terminating_sender] = False
            self.time_remaining[self.idx_terminating_sender] = 0
            self.id_data[self.idx_terminating_sender] = self.INV_ID_DATA
            self.id_receiver[self.idx_terminating_sender] = self.INV_ID_RECV
            self.payload[self.idx_terminating_sender] = self.INV_PAYLOAD
            self.mode[self.idx_terminating_sender] = self.INV_MODE
            self.length[self.idx_terminating_sender] = self.INV_LENGTH
            self.require_ack[self.idx_terminating_sender] = False
            self.rss[self.idx_terminating_sender] = NetworkMeasure.INVALID
        else:
            self.mask_active_sender[:] = False
            self.time_remaining[:] = 0
            self.id_data[:] = self.INV_ID_DATA
            self.id_receiver[:] = self.INV_ID_RECV
            self.payload[:] = self.INV_PAYLOAD
            self.mode[:] = self.INV_MODE
            self.length[:] = self.INV_LENGTH
            self.require_ack[:] = False
            self.rss[:] = NetworkMeasure.INVALID
        self.idx_terminating_sender = None
        self.set_len_cumsum()


    def __str__(self):
        # to facilitate unittest
        return (f"{'=' * len(self.__class__.__name__)}\n"
                f"{self.__class__.__name__}\n"
                f"{'-' * len(self.__class__.__name__)}\n\n"
                f"*time_remaining*\n"
                f"{self.time_remaining}\n\n"
                f"*mask_sender_step0*\n"
                f"{self.mask_sender_step0}\n\n"
                f"*mask_active_sender*\n"
                f"{self.mask_active_sender}\n\n"
                f"*len_active_sender*\n"
                f"{self.len_active_sender}\n\n"
                f"*_len_active_sender_cumsum*\n"
                f"{self._len_active_sender_cumsum}\n\n")


class EnvWirelessVector:

    def __init__(self, agents_group : dict, wifi_const):

        self.agents = agents_group
        self.num_groups = len(agents_group)
        _batch_size = np.array([v.batch_size for _, v in agents_group.items()])
        assert _batch_size.max() == _batch_size.min(), "batch size must match in different groups"
        self.batch_size = _batch_size[0]
        self.name = 'WiFi-Vector'
        self.network_measure, self.signals, self.mask_collision = {}, {}, {}
        self.rss_allpairs, self.rss_power_allpairs = {}, {}
        for name, agents in self.agents.items():
            self.network_measure[name] = NetworkMeasure(_dim1=self.batch_size, _dim2=agents.nagents)
            self.signals[name] = _Signal(_dim1=self.batch_size, _dim2=agents.nagents, T_TRANS=150)
            self.mask_collision[name] = np.ones((self.batch_size, agents.nagents, agents.nagents), dtype=np.bool)
            self.rss_allpairs[name] = None
            self.rss_power_allpairs[name] = None
            agents.set_network_measure(self.network_measure[name])
        
        self.wifi_const = wifi_const
        self.obstacles = self.wifi_const['OBSTACLES']
        
    def reset(self, obstacles, obs_attn):
        for name, agents in self.agents.items():
            agents.reset()
            self.network_measure[name].reset()
            self.signals[name].reset(hard=True)
            self.mask_collision[name][:] = True
            self.rss_allpairs[name] = None
            self.rss_power_allpairs[name] = None
            self.obstacles = obstacles
            self.obs_attn = obs_attn

    # @timeit
    def _reduceat_max_argmax(self, name_g, arr_stacked):

        len_active_sender = self.signals[name_g].len_active_sender
        len_active_sender_cumsum = self.signals[name_g]._len_active_sender_cumsum
        signals = self.signals[name_g]
        assert len(arr_stacked.shape) == 2 and len(len_active_sender.shape) == 1 and arr_stacked.shape[0] > 0
        assert arr_stacked.shape[0] == len_active_sender.sum()
        max_per_batch = np.maximum.reduceat(arr_stacked, len_active_sender_cumsum)
        arr_offset = arr_stacked - np.repeat(max_per_batch, len_active_sender, axis=0)
        idx0, idx1 = np.where(arr_offset >= 0)



        if len(idx0) > len_active_sender.size * arr_stacked.shape[1]:                     
            return None, None

        shape_argmax = (len_active_sender.size,  arr_stacked.shape[1])
        argmax_per_batch = np.zeros(shape_argmax, dtype=np.int64)
        np.put_along_axis(argmax_per_batch, idx1.reshape(shape_argmax),  idx0.reshape(shape_argmax), axis=1)
        
        _idx0 = np.arange(shape_argmax[0])[:, np.newaxis]
        _idx1 = argmax_per_batch - len_active_sender_cumsum[:, np.newaxis]
        remap_argmax = np.where(signals.mask_active_sender)[1]     
        argmax_per_batch_remap = remap_argmax[_idx1 + len_active_sender_cumsum[_idx0]]
        
        assert np.all(argmax_per_batch_remap == remap_argmax[argmax_per_batch])

        return max_per_batch, argmax_per_batch_remap
        
    @timeit
    def step0_precompute_rss(self, name_g, constants=None):

        pos = self.agents[name_g].pos
        if constants is None:           
            constants = {
                'PT': self.wifi_const['PT'], 
                'KREF': self.wifi_const['KREF'], 
                'NT': self.wifi_const['NT'], 
                'D0': self.wifi_const['D0'], 
                'OBSTACLES': self.wifi_const['OBSTACLES'], 
                'OBSTACLES_ATT': self.wifi_const['OBSTACLES_ATT']
            }
        f_euclidean = lambda a1, a2: \
            np.power(
                np.power(a1 - a2, 2).sum(axis=-1), 
                0.5
            )

        dist = f_euclidean(pos[:, np.newaxis, :, :], 
                           pos[:, :, np.newaxis, :])
        idx_zero_dist = np.where(dist == 0)
        rss = np.zeros(dist.shape)
        path_loss = -constants['KREF'] + constants['NT'] * np.log10(np.clip(dist, 1e-90, None) / constants['D0'])
        _d1, _d2, _d3 = pos.shape
        pos_from = np.broadcast_to(pos[:, :, np.newaxis, :], (_d1, _d2, _d2, _d3))
        pos_to   = np.broadcast_to(pos[:, np.newaxis, :, :], (_d1, _d2, _d2, _d3))
        pos_allpairs = np.concatenate((pos_from, pos_to), axis=-1).reshape(_d1, _d2, _d2, 2, _d3)       
        obstacle_from_PP = True
        if obstacle_from_PP:
            obstacles, obs_attn = self.obstacles, self.obs_attn
            for i in range(obstacles.shape[1]):
                is_intersect = utils.check_intersect_allpairs(pos_allpairs, [obstacles[:, i, 0], obstacles[:, i, 1], obstacles[:, i, 2], obstacles[:, i, 3]])
                
                
                _is_intersect = np.empty_like(is_intersect)
                for j in range(pos_allpairs.shape[0]):
                    _is_intersect[j] = utils.check_intersect_allpairs(pos_allpairs[j][np.newaxis, :, :, :, :], [obstacles[j, i, 0], obstacles[j, i, 1], obstacles[j, i, 2], obstacles[j, i, 3]] )
                assert np.all(is_intersect==_is_intersect)                
                path_loss += is_intersect * obs_attn[i]             
        else:
            obstacles, obs_attn = constants['OBSTACLES'], constants['OBSTACLES_ATT']
            for i, obs in enumerate(obstacles):
                is_intersect = utils.check_intersect_allpairs(pos_allpairs, [obs[0], obs[1], obs[2], obs[3]])
                path_loss += is_intersect * obs_attn[i] 
        rss = constants['PT'] - path_loss
        rss[idx_zero_dist] = constants['PT']
        self.rss_allpairs[name_g] = rss          
        self.rss_power_allpairs[name_g] = np.power(10, rss / 10)
        _perturb = self.rss_power_allpairs[name_g].min() * 1e-3
        self.rss_power_allpairs[name_g] += _perturb * np.random.uniform(low=-1, high=1, size=rss.shape)

    @timeit
    def step1_detect_collision(self, name_g, t):

        network_measure = self.network_measure[name_g]
        signals = self.signals[name_g]
        network_measure.rss[:] = -np.inf
        if signals._len_active_sender_cumsum[-1] == -1:     
            return   
        rss_power = self.rss_power_allpairs[name_g]
        mask_collision = self.mask_collision[name_g]
        assert rss_power is not None and len(rss_power.shape) == 3
        batch_size, nagents = rss_power.shape[:2]

        idx_active = np.where(signals.mask_active_sender)
        rss_power_stacked = rss_power[idx_active]
        
        rss_max_stacked, rss_argmax_orig = self._reduceat_max_argmax(name_g, rss_power_stacked)
        while rss_max_stacked is None: 
            _perturb = rss_power_stacked.min() * 1e-3
            rss_max_stacked, rss_argmax_orig = self._reduceat_max_argmax(name_g, rss_power_stacked + _perturb * 
                    np.random.uniform(low=-1, high=1, size=rss_power_stacked.shape))
        
 

        idx0_step0 = np.unique(idx_active[0])               
        idx1_step0 = rss_argmax_orig                                       
        idx2_step0 = np.arange(nagents)[np.newaxis, :]                     
        
        mask_collision_history_max = mask_collision[idx0_step0[:, np.newaxis], idx1_step0, idx2_step0].copy() 
        mask_collision[idx_active] = True                                   
        mask_collision[idx0_step0[:, np.newaxis] , idx1_step0, idx2_step0] = mask_collision_history_max    



        I_N = utils.dBmSum_reduceat(rss_power_stacked, rss_max_stacked, signals._len_active_sender_cumsum, self.wifi_const['NOISE'])
        rssi = 10 * np.log10(rss_max_stacked) - I_N         # (b, N)
        idx0_collision_cur, idx2_collision_cur = np.where(rssi < self.wifi_const['RSSI_THETA'])
        idx1_collision_cur = rss_argmax_orig[idx0_collision_cur, idx2_collision_cur]
        

        mask_collision[idx0_step0[idx0_collision_cur], idx1_collision_cur, idx2_collision_cur] = True
        
        network_measure.rss[idx0_step0] = np.add.reduceat(rss_power_stacked, signals._len_active_sender_cumsum, axis=0)

    @timeit
    def step2_wrapup_terminating_agents(self, name_g):

        signals = self.signals[name_g]
        idx_active = np.where(signals.mask_active_sender)

        assert np.all(signals.time_remaining[idx_active] > 0)
        signals.time_remaining[idx_active] -= 1
        signals.idx_terminating_sender = utils.slice_idx(idx_active, signals.time_remaining[idx_active] <= 0)

    @timeit 
    def step3_update_receiver_agents(self, name_g, t):

        signals = self.signals[name_g]
        agents = self.agents[name_g]
        network_measure = self.network_measure[name_g]
        mask_collision = self.mask_collision[name_g]
        agents.mask_channel_clear[:] = network_measure.rss < np.power(10, self.wifi_const['ED_THETA'] / 10.0)
        assert type(agents.queue_recv) == QueueHeader
        _idx, idx_success_receiver = np.where(~(mask_collision[signals.idx_terminating_sender]))        # idx into 2D
        if _idx.size == 0:
            signals.reset()
            return
        idx_success_batch = signals.idx_terminating_sender[0][_idx]
        idx_success_sender = signals.idx_terminating_sender[1][_idx]
        msg_dict = {}

        for n in agents.queue_recv.names_data_fields:
            if n == 'id_sender':
                msg_dict[n] = agents.ids[idx_success_batch, idx_success_sender]

            elif n == "rss":
                msg_dict[n] = 10 * np.log10(network_measure.rss[idx_success_batch, idx_success_receiver])   
            else:
                msg_dict[n] = getattr(signals, n)[idx_success_batch, idx_success_sender]
        agents.queue_recv.add(msg_dict, (idx_success_batch, idx_success_receiver))
        mask_collision[idx_success_batch, idx_success_sender, idx_success_receiver] = True
        signals.reset()
        
    def update(self, t, idx_send_new : Union[tuple, list], pkt_send_new : dict, is_agent_moved : bool=False):

        for name_g in self.agents.keys():
            pkts, idxs = self.agents[name_g].update(t, idx_send_new, pkt_send_new)  
            
            self.signals[name_g].add(pkts, idxs, t)
            self.mask_collision[name_g][idxs] = False  
            if is_agent_moved:
                self.step0_precompute_rss(name_g)
            self.step1_detect_collision(name_g, t)
            self.step2_wrapup_terminating_agents(name_g)
            self.step3_update_receiver_agents(name_g, t)
    
    def step(self, num_updates: int, idx_send: Union[tuple, list], pkt_send: dict, action_comm=None):

        if 'SIMPLIFIED' in self.wifi_const.keys() and self.wifi_const['SIMPLIFIED']:

            B, N = action_comm.shape[:2]
            cnt_active_sender = action_comm.sum(axis=-1)            
            cnt_active_sender_u = np.unique(cnt_active_sender)
            num_max_s = self.wifi_const['num_max_msg']

            idx_batch_sampled = []
            idx_id_sampled = []

            for cu in cnt_active_sender_u:
                idx_batch_cu = np.where(cnt_active_sender == cu)[0]
                if cu > num_max_s:
                    rng = np.random.default_rng()
                    idx_s_sampled = (rng.permuted(np.repeat(np.arange(cu)[np.newaxis, :], len(idx_batch_cu), axis=0), axis=1))[:, :num_max_s]                    
                    id_cu_stacked = np.where(action_comm[idx_batch_cu])[1].reshape(len(idx_batch_cu), cu)
                    cu_id_sampled = id_cu_stacked[np.repeat(np.arange(len(idx_batch_cu)), num_max_s), idx_s_sampled.flatten()]
                    idx_batch_sampled.append(np.repeat(idx_batch_cu, num_max_s))
                    assert len(cu_id_sampled) == len(idx_batch_cu) * num_max_s
                    idx_id_sampled.append(cu_id_sampled)
                elif cu > 0:
                    idx_batch_sampled.append(np.repeat(idx_batch_cu, cu))
                    cu_id_sampled = np.where(action_comm[idx_batch_cu])[1].flatten()
                    assert len(cu_id_sampled) == len(idx_batch_cu) * cu
                    idx_id_sampled.append(cu_id_sampled)
            if len(idx_batch_sampled) == 0:
                return np.zeros((B, N, N)), np.full((B, N, N), self.wifi_const['INVALID_RSS'])
            else:
                idx_batch_sampled = np.concatenate([x.ravel() for x in idx_batch_sampled])
                idx_id_sampled = np.concatenate([x.ravel() for x in idx_id_sampled])


            for name_g in self.agents.keys():
                assert len(list(self.agents.keys()))==1
                self.step0_precompute_rss(name_g)

                rss = self.rss_allpairs[name_g] 
                rssi =  self.rss_allpairs[name_g] - self.wifi_const['NOISE']   
                mask_rss_strong = (rssi > self.wifi_const['RSSI_THETA'])
                recv_mask = np.zeros((B, N))
                recv_mask[idx_batch_sampled, idx_id_sampled] = 1   
                recv_mask = np.repeat(recv_mask[:, np.newaxis, :], N, axis=1)
                mask_eye = np.repeat((~np.eye(N, dtype=bool))[np.newaxis, :, :], B, axis=0) 
                recv_mask *= (mask_rss_strong * mask_eye)
                recv_mask = recv_mask.astype(bool)
                if recv_mask.sum() == 0:
                    return np.zeros((B, N, N)), np.full((B, N, N), self.wifi_const['INVALID_RSS'])
                rss_ret = np.full((B, N, N), self.wifi_const['INVALID_RSS'])
                rss_ret[recv_mask] = rss[recv_mask]

                return recv_mask, rss_ret

        else:
            for t in range(num_updates):
                self.update(t, idx_send, pkt_send, is_agent_moved=(t == 0))
                idx_send, pkt_send = None, None
            meta_csr_msg, pkt_csr_msg = {}, {}
            meta_csr_ack, pkt_csr_ack = {}, {}
            for name, agents in self.agents.items():
                meta_csr_msg[name], pkt_csr_msg[name] = agents.msg_recv.get_compact_representation()
                meta_csr_ack[name], pkt_csr_ack[name] = agents.ack_recv.get_compact_representation()
            return meta_csr_msg, pkt_csr_msg, meta_csr_ack, pkt_csr_ack