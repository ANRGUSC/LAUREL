import numpy as np
from dataclasses import dataclass, fields, InitVar
from typing import List, Tuple, Union
import envs.wireless.utils as utils
from envs.games.utils import timeit

class OverflowException(Exception):
    pass

@dataclass
class NetworkMeasure:      
    """
    all have shape (batch, nagents). network measure will be overwritten everytime an agent
    receives a new ack or msg. 
    """
    # data fields
    rss         : np.ndarray = np.array([])
    # book-keeper
    # constants
    INVALID = -np.inf
    # init fields
    _dim1       : InitVar[int] = 0
    _dim2       : InitVar[int] = 0
    # summary
    names_fields = ['rss']

    def __post_init__(self, _dim1, _dim2):
        if _dim1 > 0 and _dim2 > 0:
            for n in self.names_fields:
                setattr(self, n, np.full((_dim1, _dim2), self.INVALID))

    def reset(self):
        for n in self.names_fields:
            getattr(self, n)[:] = self.INVALID

@dataclass
class Buffer:

    # data fields
    id_sender           : np.ndarray = np.array([])        
    id_data             : np.ndarray = np.array([])        
    rss                 : np.ndarray = np.array([])         
    # book-keeper
    _idx_next           : np.ndarray = np.array([])        
    _idx_max            : int = 0                          
    _capacity           : int = 0
    # constant
    INITIAL  = -1            
    INVALID  = -2           
    # init fields
    _dim1               : InitVar[int] = 0
    _dim2               : InitVar[int] = 0
    _dim3               : InitVar[int] = 0
    # summary
    names_data_fields = ['id_sender', 'id_data', 'rss']
    def __post_init__(self, _dim1 : int, _dim2 : int, _dim3 : int):
        if _dim1 > 0 and _dim2 > 0 and _dim3 > 0:
            self.id_sender = np.full((_dim1, _dim2, _dim3), self.INITIAL, dtype=np.int64)
            self.id_data  = np.full((_dim1, _dim2, _dim3), self.INITIAL, dtype=np.int64)
            self.rss = np.full((_dim1, _dim2, _dim3), self.INITIAL)
            self.capacity = _dim3
        else:
            assert self.id_sender.shape == self.id_data.shape == self.rss.shape
            _dim1, _dim2, _dim3 = self.id_sender.shape
            self.capacity = self.id_sender.shape[-1]
        self._idx_next = np.zeros((_dim1, _dim2), dtype=np.int64)
        self._idx_max = 0
        self._capacity = _dim3

    # @timeit
    def add(self, idx_new : Union[tuple, list], data_new : dict):

        assert len(idx_new) == 2 and idx_new[0].shape == idx_new[1].shape
        assert len(data_new['id_sender'].shape) == 1 and data_new['id_sender'].shape == data_new['id_data'].shape == data_new['rss'].shape
        if self._idx_max > 0:        
            id_sender_slice = self.id_sender[idx_new][:, :self._idx_max]
            id_data_slice   = self.id_data[idx_new][:, :self._idx_max]
            _idx = np.where((id_sender_slice == data_new['id_sender'][:, np.newaxis])
                          * (id_data_slice == data_new['id_data'][:, np.newaxis]))
            for n in self.names_data_fields:
                data_new[n][_idx[0]] = self.INVALID
        idxRel_add = np.where((data_new['id_sender'] != self.INVALID) * (data_new['id_data'] != self.INVALID))        
        idx0_add = idx_new[0][idxRel_add]
        idx1_add = idx_new[1][idxRel_add]
        idx2_add = self._idx_next[idx0_add, idx1_add].flatten()
        if idx0_add.size:    
            self._idx_max = max(np.max(idx2_add) + 1, self._idx_max)
        if self._idx_max > self.id_sender.shape[-1]:
            raise OverflowException
        for n in self.names_data_fields:
            getattr(self, n)[idx0_add, idx1_add, idx2_add] = data_new[n][idxRel_add]
        self._idx_next[idx0_add, idx1_add] += 1
        return idx0_add, idx1_add

    # @timeit
    def remove(self, idx_batch : np.ndarray, idx_agent : np.ndarray, idx_depth : np.ndarray):

        if idx_batch.size == 0:
            return
        idx_next = self._idx_next[idx_batch, idx_agent]
        assert np.all(idx_next > 0), "cannot remove entry from empty buffer(s). "
        assert idx_next.shape == idx_depth.shape and np.all(idx_next > idx_depth)
        for fn in self.names_data_fields:
            data_orig = getattr(self, fn)[idx_batch, idx_agent, :self._idx_max]
            data_shift = np.roll(data_orig, -1, axis=-1)
            mask = np.arange(self._idx_max)[np.newaxis, :] >= idx_depth[:, np.newaxis]
            data_new = ~mask * data_orig + mask * data_shift
            data_new[:, -1] = self.INVALID
            getattr(self, fn)[idx_batch, idx_agent, :self._idx_max] = data_new
        self._idx_next[idx_batch, idx_agent] -= 1
        if self._idx_next[idx_batch, idx_agent].max() + 1 == self._idx_max:
            self._idx_max = self._idx_next.max()
    
    # @timeit
    def remove_all(self, idx_batch : np.ndarray, idx_agent : np.ndarray):
        if idx_batch.size == 0:
            return
        idx_next_slice = self._idx_next[idx_batch, idx_agent]
        idx_batch_re = np.repeat(idx_batch, idx_next_slice)
        idx_agent_re = np.repeat(idx_agent, idx_next_slice)
        idx_size_re  = utils.generate_piecewise_arange(idx_next_slice)
        for fn in self.names_data_fields:
            getattr(self, fn)[idx_batch_re, idx_agent_re, idx_size_re] = self.INVALID
        if self._idx_next[idx_batch, idx_agent].max() == self._idx_max:
            self._idx_next[idx_batch, idx_agent] = 0
            self._idx_max = self._idx_next.max()        
        else:
            self._idx_next[idx_batch, idx_agent] = 0

    def get_idx_id_match(self, idx : Union[tuple, list], id_sender_candy : np.ndarray, id_data_candy : np.ndarray):

        assert len(id_sender_candy.shape) == len(id_data_candy.shape) == 1
        mask = (self.id_sender[idx][:, :self._idx_max] == id_sender_candy[:, np.newaxis]) \
             * (self.id_data[idx][:, :self._idx_max] == id_data_candy[:, np.newaxis])
        assert len(mask.shape) == 2
        _idx0, _idx1 = np.where(mask)
        idx_ret = utils.slice_idx(idx, _idx0)
        assert len(idx_ret) == 2 and idx_ret[0].size == _idx1.size
        return idx_ret[0], idx_ret[1], _idx1

    def reset(self):
        for n in self.names_data_fields:
            getattr(self, n)[:] = self.INITIAL
        self._idx_next[:] = 0
        self._idx_max = 0

    def get_compact_representation(self, copy=True):
 
        _idx0, _idx1 = np.where(self._idx_next > 0)
        if _idx0.size == 0:
            return {"indptr": np.array([0], dtype=np.int), 'idx_batch': np.array([]), 'idx_agent': np.array([])}, \
                   {n: np.array([]) for n in self.names_data_fields}

        idx0 = np.repeat(_idx0, self._idx_next[_idx0, _idx1])

        idx1 = np.repeat(_idx1, self._idx_next[_idx0, _idx1])
        idx2 = utils.generate_piecewise_arange(self._idx_next.flatten())
        pkt_csr = {}
        for n in self.names_data_fields:
            if copy:
                pkt_csr[n] = getattr(self, n)[idx0, idx1, idx2].copy()
            else:
                pkt_csr[n] = getattr(self, n)[idx0, idx1, idx2]
        indptr = np.zeros(_idx0.size + 1, dtype=np.int)
        indptr[1:] = np.cumsum(self._idx_next[_idx0, _idx1])
        meta_csr = {"indptr"    : indptr,
                    "idx_batch" : _idx0,
                    "idx_agent" : _idx1}
        return meta_csr, pkt_csr

    def __str__(self):
        return (f"id_sender\n"
                f"-------------\n"
                f"{self.id_sender}\n"
                f"id_data\n"
                f"-------------\n"
                f"{self.id_data}\n\n"
                f"_idx_next\n"
                f"---------\n"
                f"{self._idx_next}\n\n"
                f"_idx_max\n"
                f"-------\n"
                f"{self._idx_max}\n")



@dataclass
class QueueHeader:

    id_sender           : np.ndarray = np.array([])   
    id_data             : np.ndarray = np.array([])
    id_receiver         : np.ndarray = np.array([])
    payload             : np.ndarray = np.array([])     
    mode                : np.ndarray = np.array([])    
    length              : np.ndarray = np.array([])    
    require_ack         : np.ndarray = np.array([])
    rss                 : np.ndarray = np.array([])
    # book-keeper
    _ptr_start          : np.ndarray = np.array([])
    _ptr_end            : np.ndarray = np.array([])
    _depth              : np.ndarray = np.array([])
    _idx_valid_data     : Tuple[np.ndarray] = None     
    _capacity           : int = 0
    # constants
    INVALID = -1
    # init fields
    _dim1               : InitVar[int] = 0
    _dim2               : InitVar[int] = 0
    _dim3               : InitVar[int] = 0
    # summary
    names_data_fields = ['id_sender', 'id_data', 'id_receiver', 'payload', 'mode', 'length', 'require_ack', 'rss']
    
    def __post_init__(self, _dim1 : int, _dim2 : int, _dim3 : int):
        if _dim1 > 0 and _dim2 > 0 and _dim3 > 0:
            self.id_sender   = np.zeros((_dim1, _dim2, _dim3), dtype=np.int64)
            self.id_data     = np.zeros((_dim1, _dim2, _dim3), dtype=np.int64)
            self.id_receiver = np.zeros((_dim1, _dim2, _dim3), dtype=np.int64)
            self.payload     = np.zeros((_dim1, _dim2, _dim3))
            self.mode        = np.zeros((_dim1, _dim2, _dim3))
            self.length      = np.zeros((_dim1, _dim2, _dim3))
            self.require_ack = np.zeros((_dim1, _dim2, _dim3), dtype=np.bool)
            self.rss         = np.zeros((_dim1, _dim2, _dim3))
            self._idx_valid_data = (np.array([]), np.array([]))
        self._ptr_start = np.zeros((self.id_sender.shape[0], self.id_sender.shape[1]), dtype=np.int32)
        self._ptr_end   = np.zeros((self.id_sender.shape[0], self.id_sender.shape[1]), dtype=np.int32)
        self._depth = np.zeros(self._ptr_start.shape, dtype=np.int32)
        self._capacity = self.id_sender.shape[-1]

    # @timeit
    def add(self, msg_dict : dict, idx : Union[tuple, list]):

        assert type(msg_dict) == dict
        assert np.all(self._depth[idx] <= self._capacity - 1), "FIFO overflow!"
        assert len(idx) == 2 and idx[0].size == idx[1].size
        idx2 = self._ptr_end[idx]
        for fname in self.names_data_fields:
            getattr(self, fname)[idx[0], idx[1], idx2] = msg_dict[fname]
        self._depth[idx] += 1
        self._ptr_end[idx] = np.mod(idx2 + 1, self._capacity)

    # @timeit
    def add_to_front(self, msg_dict : dict, idx : Union[tuple, list]):

        assert type(msg_dict) == dict
        assert np.all(self._depth[idx] <= self._capacity - 1)
        assert len(idx) == 2 and idx[0].size == idx[1].size
        idx2 = np.mod(self._ptr_start[idx] - 1, self._capacity)
        self._ptr_start[idx] = idx2
        for fname in self.names_data_fields:
            getattr(self, fname)[idx[0], idx[1], idx2] = msg_dict[fname]
        self._depth[idx] += 1

    # @timeit 
    def retrieve(self, idx : Union[tuple, list], names_fields : Union[list, str]=None, copy : bool=True):

        assert len(idx) == 2 and idx[0].size == idx[1].size
        assert np.all(self._depth[idx] >= 1), "some FIFO queue(s) is empty. Cannot retrieve!"
        names_fields = self.names_data_fields if names_fields is None else names_fields
        idx2 = self._ptr_start[idx]
        if type(names_fields) == str:
            ret = getattr(self, names_fields)[idx[0], idx[1], idx2]
            return ret.copy() if copy else ret
        elif copy:
            return {n: getattr(self, n)[idx[0], idx[1], idx2].copy() for n in names_fields}
        else:
            return {n: getattr(self, n)[idx[0], idx[1], idx2] for n in names_fields}

    def remove(self, idx : Union[tuple, list]):

        assert len(idx) == 2 and idx[0].size == idx[1].size
        assert np.all(self._depth[idx] >= 1), "some FIFO queue(s) is empty. Cannot retrieve!"
        self._depth[idx] -= 1
        self._ptr_start[idx] = np.mod(self._ptr_start[idx] + 1, self._capacity)

    def pop(self, idx : Union[tuple, list], copy : bool=True):
        ret = self.retrieve(idx, copy=copy)
        self.remove(idx)
        return ret

    def reset(self):
        for n in self.names_data_fields:
            getattr(self, n)[:] = self.INVALID
        self._ptr_start[:] = 0
        self._ptr_end[:]   = 0
        self._depth[:]     = 0

