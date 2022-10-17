import numpy as np
from dataclasses import dataclass, field, fields, InitVar
from typing import List, Tuple, Union
from envs.wireless.protocols import Protocols
from envs.wireless.data_struct import NetworkMeasure, Buffer, QueueHeader
import envs.wireless.utils as utils
from collections import defaultdict
import math
from envs.games.utils import timeit



class pCSMAs(Protocols):
    def __init__(self, ids_agent : np.ndarray, wifi_const, p : int=-1):
        super().__init__(ids_agent, wifi_const)
        self.p = p if p > 0 else wifi_const['P']
        self.WIN_CONTENTION = wifi_const['CONTENTION_WINDOW']
        self.TIME_WAIT_ACK = wifi_const['ACK_WAIT']
        self.RATIO_RX = wifi_const['Rx_R']
        self.NUM_MAX_RETRANS = wifi_const['RETXMAX']
        self.LEN_MSG = wifi_const['PKT_TIME']
        self.LEN_ACK = wifi_const['ACK_PKT_TIME']   
        self._counter_wait_ack = np.zeros((self.para_vec, self.nagents), dtype=np.int)
        self._counter_backoff = np.zeros((self.para_vec, self.nagents), dtype=np.int)
        self._counter_reTx = np.zeros((self.para_vec, self.nagents), dtype=np.int)
        self.WAITING_FOR_ACK = 0
        self.QUEUE_IS_EMPTY  = 1
        self.OUTGOING_MESSAGE_PENDING = 2

        self.threshold_num_unrecv = math.ceil((self.nagents - 1) * (1 - self.RATIO_RX))

    def reset(self):
        super().reset()
        self._state_agents[:] = self.QUEUE_IS_EMPTY
        self._counter_wait_ack[:] = 0
        self._reset_backoff_counter(None)
        self._counter_reTx[:] = 0
        
    @timeit  
    def step0_receive(self, t):

        queue = self.queue_recv                                                                
        idx_all = np.where(queue._depth)                                                       
        if idx_all[0].size == 0:
            return {'send': [(np.array([], dtype=np.int64), np.array([], dtype=np.int64))], 'cleanup': [idx_all]}
        pkt_info = queue.retrieve(idx_all, names_fields=['id_receiver', 'mode', 'id_sender'], copy=False)
        idx_recv = utils.slice_idx(idx_all,                                                    
                        (pkt_info['id_receiver'] == self.ids[idx_all])                        
                      + (pkt_info['id_receiver'] == self.ID_BROADCAST) * (pkt_info['id_sender'] != idx_all[1])                 
                      + (pkt_info['mode'] == self.MODE_BROADCAST) * (pkt_info['id_sender'] != idx_all[1])                       
                    )                                                                           
        type_payload = queue.retrieve(idx_recv, names_fields="payload", copy=False)
        idx_ack = utils.slice_idx(idx_recv, type_payload == self.PAYLOAD_ACK)                  
        idx_msg = utils.slice_idx(idx_recv, type_payload == self.PAYLOAD_MSG)                 
        if idx_msg[0].size == 0:
            return {'send': [idx_ack], 'cleanup': [idx_all]}
        mask_req_ack = queue.retrieve(idx_msg, names_fields='require_ack', copy=False)         
        idx_new_ack = utils.slice_idx(idx_msg, mask_req_ack)                                   
        pkt_new_ack = self.prepare_ack_pkt(idx_new_ack, copy=False)                          
        self.queue_send_ack.add(pkt_new_ack, idx_new_ack)                                      
        self.buffer_recv_msg.add(idx_msg, queue.retrieve(idx_msg, names_fields=Buffer.names_data_fields, copy=False))       
        assert idx_ack[0].size + idx_msg[0].size == idx_recv[0].size, "Protocols seem to encounter payload other than ACK and MSG! "
        return {"send": [idx_ack], "cleanup": [idx_all]}

    def step1_send(self, t, idx_pkt_ack : Union[tuple, list]):
        """ EQUIVALENT OF _handle_outgoing_packets()
        """
        idx_empty    = np.where(self._state_agents == self.QUEUE_IS_EMPTY)                        
        idx_pending  = np.where(self._state_agents == self.OUTGOING_MESSAGE_PENDING)              
        idx_wait_ack = np.where(self._state_agents == self.WAITING_FOR_ACK)                         
        assert idx_empty[0].size + idx_pending[0].size + idx_wait_ack[0].size \
                == np.product(self._state_agents.shape), "There is unknown state!"                 
        
        idx_send = utils.slice_idx(idx_empty, ~self._mask_empty_queue_send(idx_empty))              
        self._state_agents[idx_send] = self.OUTGOING_MESSAGE_PENDING                               
        pkt2envs, idx2envs = self._outgoing_msg_pending(t, utils.merge_idx(idx_send, idx_pending))  
        self._waiting_for_ack(t, idx_wait_ack, idx_pkt_ack)                                         
        return pkt2envs, idx2envs, []

    def cleanup(self, t, idx_recv_queue):
        self.queue_recv.remove(idx_recv_queue)

    def _reset_backoff_counter(self, idx):
        if idx is not None:
            assert len(idx) == 2
            self._counter_backoff[idx] = np.random.randint(0, self.WIN_CONTENTION, idx[0].size)
        else:
            self._counter_backoff = np.random.randint(0, self.WIN_CONTENTION, self._counter_backoff.shape)

    @timeit
    def _outgoing_msg_pending(self, t, idx : Union[tuple, list]):
        mask_timeup = self._counter_backoff[idx] <= 0                                               
        idx_append_candy = utils.slice_idx(idx, mask_timeup)                                       
        self._counter_backoff[idx] -= ~mask_timeup                                                  
        mask_clear = self.mask_channel_clear[idx_append_candy]                                      
        idx_busy_timeup  = utils.slice_idx(idx_append_candy, ~mask_clear)                           
        self._reset_backoff_counter(idx_busy_timeup)                                                
        idx_clear_timeup = utils.slice_idx(idx_append_candy, mask_clear)                            
        mask_contention = np.random.rand(idx_clear_timeup[0].size) < self.p                          
        idx_lucky = utils.slice_idx(idx_clear_timeup, mask_contention)                             
        mask_ack_non0 = ~self._mask_empty_queue_send(idx_lucky, name_queue='ack')                   
        idx_ack = utils.slice_idx(idx_lucky, mask_ack_non0)                                         
        idx_msg = utils.slice_idx(idx_lucky, ~mask_ack_non0)                                       
        if idx_ack[0].size > 0:
            pkt_ack = self.queue_send_ack.pop(idx_ack, copy=(idx_msg[0].size == 0))                                                 
        else:
            pkt_ack = None
        if idx_msg[0].size > 0:
            pkt_msg = self.queue_send.retrieve(idx_msg, copy=(idx_ack[0].size == 0))                        
            idx_msg_to_ack = utils.slice_idx(idx_msg, pkt_msg['require_ack'])                           
            pkt_msg_to_ack = utils.slice_pkt(pkt_msg, pkt_msg['require_ack'])                           
            if idx_msg_to_ack[0].size > 0:
                self._add_expected_ack(pkt_msg_to_ack, idx_msg_to_ack, t)                                     
            self._counter_wait_ack[idx_msg_to_ack] = self.TIME_WAIT_ACK                                 
            self._state_agents[idx_msg_to_ack] = self.WAITING_FOR_ACK                                 
            idx_msg_to_remove = utils.slice_idx(idx_msg, ~pkt_msg['require_ack'])                      
            self.queue_send.remove(idx_msg_to_remove)                                                  
        else:
            idx_msg_to_remove = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
            pkt_msg = None
        if idx_ack[0].size > 0:
            idx_trans_candy = utils.merge_idx(idx_ack, idx_msg_to_remove)                               
            mask_trans_candy = self._mask_empty_queue_send(idx_trans_candy)                             # ^
            self._state_agents[utils.slice_idx(idx_trans_candy, mask_trans_candy)] = self.QUEUE_IS_EMPTY  # ^

            self._reset_backoff_counter(utils.slice_idx(idx_trans_candy, mask_trans_candy))

        return utils.merge_dict(pkt_ack, pkt_msg), utils.merge_idx(idx_ack, idx_msg)

    @timeit
    def _waiting_for_ack(self, t, idx_state_ack : Union[tuple, list], idx_pkt_ack : Union[tuple, list]):
        if idx_state_ack[0].size == 0:
            return
        _mask_expected = self.buffer_expected_ack._idx_next[idx_state_ack] > 0
        _mask_counter  = self._counter_wait_ack[idx_state_ack] > 0
        idx_exp1_cnt1 = utils.slice_idx(idx_state_ack, _mask_expected * _mask_counter)           
        self._counter_wait_ack[idx_exp1_cnt1] -= 1                                                 
        if idx_pkt_ack[0].size > 0:
            idx_ack = self._intersect(idx_pkt_ack, idx_exp1_cnt1)                                       
            pkt_ack = self.queue_recv.retrieve(idx_ack, names_fields=['id_sender', 'id_data'], copy=False)  
            if self.buffer_expected_ack._idx_max > 0:
                _idx0, _idx1, _idx2 = self.buffer_expected_ack.get_idx_id_match(idx_ack, pkt_ack["id_sender"], pkt_ack["id_data"])    
                pkt_match = self.queue_recv.retrieve((_idx0, _idx1), names_fields=Buffer.names_data_fields, copy=False)
                self.buffer_recv_ack.add((_idx0, _idx1), pkt_match)                                    
                self.buffer_expected_ack.remove(_idx0, _idx1, _idx2)                                   
        if _mask_counter.sum() < _mask_counter.size:
            idx_exp1_cnt0 = utils.slice_idx(idx_state_ack,  _mask_expected * ~_mask_counter)            
            mask_end_trans = (self._counter_reTx[idx_exp1_cnt0] >= self.NUM_MAX_RETRANS) \
                        + (self.buffer_expected_ack._idx_next[idx_exp1_cnt0] <= self.threshold_num_unrecv)
            idx_end_trans = utils.slice_idx(idx_exp1_cnt0, mask_end_trans)                              
            self.buffer_expected_ack.remove_all(*idx_end_trans)                                        
            self._counter_reTx[idx_end_trans] = 0                                                      
            idx_keep_trans = utils.slice_idx(idx_exp1_cnt0, ~mask_end_trans)                            
            self._counter_reTx[idx_keep_trans] += 1                                                     
            self._state_agents[idx_keep_trans] = self.OUTGOING_MESSAGE_PENDING                          
        else:
            idx_keep_trans = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        if _mask_expected.sum() < _mask_expected.size:
            idx_exp0 = utils.slice_idx(idx_state_ack, ~_mask_expected)                           
            self._counter_reTx[idx_exp0] = 0                                                            
            self._counter_wait_ack[idx_exp0] = 0                                                      
            self.queue_send.remove(idx_exp0)                                                          
            self._state_agents[idx_exp0] = self.QUEUE_IS_EMPTY                                          
            self._state_agents[
                    utils.slice_idx(idx_exp0, ~self._mask_empty_queue_send(idx_exp0))
                ] = self.OUTGOING_MESSAGE_PENDING                                                      
            self._reset_backoff_counter(utils.merge_idx(idx_keep_trans, idx_exp0))                      
        
    def _add_expected_ack(self, pkt_new : dict, idx_new : Union[tuple, list], t):

        buffer = self.buffer_expected_ack
        _buffer2check = buffer.id_data[idx_new[0], idx_new[1], :buffer._idx_max]                                   
        assert len(pkt_new['id_data'].shape)==1, "one agent is sending multiple msg pkts"
        mask_added = np.repeat(pkt_new['id_data'][:, np.newaxis], buffer._idx_max, axis=-1)==_buffer2check
        idx_new = utils.slice_idx(idx_new, ~np.any(mask_added, axis=-1))

        assert np.all(self._counter_reTx[idx_new] == 0)                                                             
        pkt_new = self.queue_send.retrieve(idx_new, names_fields=['mode', 'id_receiver', 'id_data'], copy=False)   
        mask_broadcast = pkt_new['mode'] == self.MODE_BROADCAST
        idx_broadcast = utils.slice_idx(idx_new, mask_broadcast)
        arange = np.arange(self.nagents - 1)
        _idx0 = idx_broadcast[0][:, np.newaxis]
        _idx1 = idx_broadcast[1][:, np.newaxis]
        _idx2 = arange[np.newaxis, :] + buffer._idx_next[idx_broadcast][:, np.newaxis]
        _idx_val = arange[np.newaxis, :] + (arange[np.newaxis, :] >= _idx1).astype(arange.dtype)
        buffer.id_sender[_idx0, _idx1, _idx2] = self.ids[_idx0, _idx_val]
        buffer.id_data[_idx0, _idx1, _idx2] = pkt_new['id_data'][mask_broadcast][:, np.newaxis]
        buffer._idx_next[_idx0, _idx1] += self.nagents - 1
        if _idx0.shape[0] > 0:
            buffer._idx_max = max(buffer._idx_max, np.max(buffer._idx_next[_idx0, _idx1]))

        idxRel_unicast = np.where(pkt_new['mode'] == self.MODE_UNICAST)
        _idx0, _idx1 = utils.slice_idx(idx_new, idxRel_unicast)
        _idx2 = buffer._idx_next[_idx0, _idx1]
        buffer.id_sender[_idx0, _idx1, _idx2] = pkt_new['id_receiver'][idxRel_unicast]
        buffer.id_data[_idx0, _idx1, _idx2] = pkt_new['id_data'][idxRel_unicast]
        buffer._idx_next[_idx0, _idx1] += 1
        if _idx0.shape[0] > 0:
            buffer._idx_max = max(buffer._idx_max, np.max(buffer._idx_next[_idx0, _idx1]))
        assert buffer._idx_max <= buffer.capacity, "Buffer overflow for expected_ack! "

    def _mask_empty_queue_send(self, idx : Union[tuple, list], name_queue : str=None):
        if name_queue is None or name_queue == ['ack', 'msg'] or name_queue == ['msg', 'ack']:
            return (self.queue_send._depth[idx] == 0) * (self.queue_send_ack._depth[idx] == 0)
        elif name_queue == 'ack':
            return self.queue_send_ack._depth[idx] == 0
        elif name_queue == 'msg':
            return self.queue_send._depth[idx] == 0
        else:
            raise NotImplementedError

    def prepare_ack_pkt(self, idx_new_ack : Union[tuple, list], copy : bool=True):
        """
        NOTE that an agent cannot receive multiple msg at the same time
        """
        pkt_new_ack = self.queue_recv.retrieve(idx_new_ack, names_fields=['id_sender', 'id_data', 'id_receiver'], copy=copy)
        _shape_pkt = pkt_new_ack['id_sender'].shape
        pkt_new_ack['id_receiver'] = pkt_new_ack['id_sender']
        pkt_new_ack['id_sender']   = idx_new_ack[1]                                    
        pkt_new_ack['payload']     = np.broadcast_to(self.PAYLOAD_ACK, _shape_pkt)
        pkt_new_ack['mode']        = np.broadcast_to(self.MODE_UNICAST, _shape_pkt)
        pkt_new_ack['length']      = np.broadcast_to(self.LEN_ACK, _shape_pkt)
        pkt_new_ack['require_ack'] = np.broadcast_to(False, _shape_pkt)
        if not copy:
            pkt_new_ack['rss'] = self.measure.rss[idx_new_ack]
        else:
            pkt_new_ack['rss'] = self.measure.rss[idx_new_ack].copy()
        return pkt_new_ack

    def prepare_msg_pkt(self, idx_new_msg : Union[tuple, list], copy : bool=True):

        pass