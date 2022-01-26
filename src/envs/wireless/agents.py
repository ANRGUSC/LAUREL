from envs.wireless.protocols import QueueHeader, Buffer
from envs.wireless.pcsma import pCSMAs
import os
import time
import numpy as np
from dataclasses import dataclass, field, fields, InitVar
from collections import defaultdict
from typing import List, Tuple, Union
from envs.games.utils import timeit


class Agents:
    """
    Batch processing of all agents, via numpy ndarray representation
    Dimensions: 
        batch x nagents x ...
    """
    def __init__(self, ids_agent, wifi_const, can_comm=True, log_event=False):
        assert len(ids_agent.shape) == 2
        batch_size, nagents = ids_agent.shape
        self.batch_size = batch_size
        self.nagents = nagents
        self.pos = np.zeros((batch_size, nagents, 2))       # last dim of 2 = x, y
        self.mask_freeze = np.zeros((batch_size, nagents), dtype=np.bool)       # interface with PP
        self.ids = ids_agent
        if can_comm:
            self.name_protocol = 'pCSMA'
            self.protocols = pCSMAs(self.ids, wifi_const)

            self.msg_recv = self.protocols.buffer_recv_msg
            self.ack_recv = self.protocols.buffer_recv_ack
            self.queue_recv = self.protocols.queue_recv
            self.mask_channel_clear = self.protocols.mask_channel_clear

        self.log_event = log_event
        if self.log_event:
            self.logger_idx_next_msg = np.zeros((batch_size, nagents), dtype=np.int)
            self.logger_idx_next_ack = np.zeros((batch_size, nagents), dtype=np.int)

        self.output_files = None

    def set_network_measure(self, measure):
        """ Interface with envs """
        self.protocols.measure = measure

    def reset(self):
        """
        Reset buffer for received data
        """
        self.protocols.reset()

    # @timeit
    def update(self, t, idx_new_send : Union[tuple, list]=None, pkt_new_send : dict=None):
        """
        The pkt_new_send can only be a msg. As acks are auto-generated within the protocol. 
        """
        if idx_new_send is not None and pkt_new_send is not None:
            assert len(idx_new_send) == 2 and type(pkt_new_send) == dict
            self.protocols.queue_send.add(pkt_new_send, idx_new_send)
        # protocol update will return the pkts to be propagated in the wireless env at the current time t. 
        pkts2envs, idxs2envs = self.protocols.update(t)
        
        if self.log_event:
            self.log_events(t, *self._get_new_pkt())
        
        return pkts2envs, idxs2envs

    def set_output_file(self, output_file):
        # create folder first
        output_dir = output_file.split("/")
        if len(output_dir) > 1:
            output_dir = '/'.join(output_dir[:-1])
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        # setup file names
        self.output_files = [f"{output_file}_batch{i}.log" for i in range(self.batch_size)]
    
    def _get_new_pkt(self):
        """
        Return the new pkts and their idxs compared with the previous time step
        """
        idx_new_msg = np.where(self.logger_idx_next_msg < self.msg_recv._idx_next)
        idx_new_ack = np.where(self.logger_idx_next_ack < self.ack_recv._idx_next)
        idx0_msg, idx1_msg = idx_new_msg
        idx2_msg = self.msg_recv._idx_next[idx_new_msg] - 1
        idx0_ack, idx1_ack = idx_new_ack
        idx2_ack = self.ack_recv._idx_next[idx_new_ack] - 1
        pkt_new_msg = {"id_sender"      : self.msg_recv.id_sender[idx0_msg, idx1_msg, idx2_msg],
                    "id_data"        : self.msg_recv.id_data[idx0_msg, idx1_msg, idx2_msg],
                    "id_receiver"    : self.ids[idx_new_msg]}
        pkt_new_ack = {"id_sender"      : self.ack_recv.id_sender[idx0_ack, idx1_ack, idx2_ack],
                    "id_data"        : self.ack_recv.id_data[idx0_ack, idx1_ack, idx2_ack],
                    "id_receiver"    : self.ids[idx_new_ack]}
        self.logger_idx_next_msg[:] = self.msg_recv._idx_next[:]
        self.logger_idx_next_ack[:] = self.ack_recv._idx_next[:]
        return idx_new_msg, pkt_new_msg, idx_new_ack, pkt_new_ack

    def log_events(self, t, idx_new_msg, pkt_new_msg, idx_new_ack, pkt_new_ack):
        """
        Log who gets new pkt for all agents in all batches. 
        """
        if self.output_files:
            # The string to write to each log file
            log_batch = defaultdict(lambda : defaultdict(list))
            idx_new_msg_stack = np.vstack(idx_new_msg).T
            assert idx_new_msg_stack.shape[0] == pkt_new_msg['id_data'].size
            assert pkt_new_msg['id_sender'].size == pkt_new_msg['id_data'].size == pkt_new_msg['id_receiver'].size
            for i, (ib, ia) in enumerate(idx_new_msg_stack):        # ib: batch index;  ia: agent index
                log_batch[ib][ia].append((f"[AGENT {pkt_new_msg['id_receiver'][i]}] (time {t:4d}) "
                                          f"receive MSG {pkt_new_msg['id_data'][i]} "
                                          f"from Agent {pkt_new_msg['id_sender'][i]}"))
            idx_new_ack_stack = np.vstack(idx_new_ack).T
            assert idx_new_ack_stack.shape[0] == pkt_new_ack['id_data'].size
            assert pkt_new_ack['id_sender'].size == pkt_new_ack['id_data'].size == pkt_new_ack['id_receiver'].size
            for i, (ib, ia) in enumerate(idx_new_ack_stack):        # ib: batch index;  ia: agent index
                log_batch[ib][ia].append((f"[AGENT {pkt_new_ack['id_receiver'][i]}] (time {t:4d}) "
                                          f"receive ACK {pkt_new_ack['id_data'][i]} "
                                          f"from Agent {pkt_new_ack['id_sender'][i]}"))
            for ib, mb_ag in log_batch.items():
                with open(self.output_files[ib], 'a') as fb:
                    fb.write('\n'.join('\n'.join(msg_l) for ia, msg_l in mb_ag.items()))
