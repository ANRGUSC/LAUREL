import numpy as np
from dataclasses import dataclass, field, fields, InitVar
from typing import List, Tuple, Union
import envs.wireless.utils as utils
from collections import defaultdict
from envs.wireless.data_struct import NetworkMeasure, Buffer, QueueHeader
import time


class Protocols:
    def __init__(self, ids_agent : np.ndarray, wifi_const):
        self.INVALID_STATE = -1
        self.ID_BROADCAST = 10000
        self.MODE_BROADCAST = 0
        self.MODE_UNICAST   = 1
        self.PAYLOAD_ACK = 0
        self.PAYLOAD_MSG = 1
        self.ids = ids_agent
        max_msg_per_agent = wifi_const['MAX_MSG_PER_AGENT']
        self.para_vec, self.nagents = ids_agent.shape
        self.queue_recv     = QueueHeader(_dim1=self.para_vec, _dim2=self.nagents, _dim3=5)                                      
        self.queue_send     = QueueHeader(_dim1=self.para_vec, _dim2=self.nagents, _dim3=max_msg_per_agent)
        self.queue_send_ack = QueueHeader(_dim1=self.para_vec, _dim2=self.nagents, _dim3=max_msg_per_agent * (self.nagents - 1)) 
        self.buffer_recv_msg = Buffer(_dim1=self.para_vec, _dim2=self.nagents, _dim3=max_msg_per_agent * (self.nagents - 1))
        self.buffer_recv_ack = Buffer(_dim1=self.para_vec, _dim2=self.nagents, _dim3=max_msg_per_agent * (self.nagents - 1))
        self.buffer_expected_ack = Buffer(_dim1=self.para_vec, _dim2=self.nagents, _dim3=max_msg_per_agent * (self.nagents - 1))
        self.measure = None    
        self.mask_channel_clear = np.ones((self.para_vec, self.nagents), dtype=np.bool)
        self._state_agents = np.full((self.para_vec, self.nagents), self.INVALID_STATE)          
        self.placeholder = np.zeros((self.para_vec, self.nagents), dtype=np.bool)
                

    def reset(self):
        self.queue_recv.reset()
        self.queue_send.reset()
        self.queue_send_ack.reset()
        self.buffer_recv_msg.reset()
        self.buffer_recv_ack.reset()
        self.buffer_expected_ack.reset()
        self.mask_channel_clear[:] = True
        

    def _intersect(self, idx1, idx2):
        assert len(idx1) == len(idx2) == 2
        self.placeholder[idx1] = 1
        ret = utils.slice_idx(idx2, self.placeholder[idx2])
        self.placeholder[idx1] = 0
        return ret

    def step0_receive(self, t):
        raise NotImplementedError

    def step1_send(self, t, *args):
        raise NotImplementedError

    def cleanup(self, t, *args):
        raise NotImplementedError

    def update(self, t):

        ret_recv = self.step0_receive(t)
        pkts, idxs, args_cleanup = self.step1_send(t, *ret_recv['send'])
        self.cleanup(t, *ret_recv['cleanup'], *args_cleanup)
        return pkts, idxs

