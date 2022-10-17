import numpy as np
import math
from collections import defaultdict
from typing import Union
from envs.games.utils import timeit

def generate_piecewise_arange(length):
    """
    Generate the number pattern of piecewise arange. 
    e.g., length = 2, 3, 1, 4, 0, 1
    Then the generated pattern would be:
        0, 1, 0, 1, 2, 0, 0, 1, 2, 3, 0
    """
    assert type(length) == np.ndarray
    if length.size == 0:
        return np.array([], dtype=np.int64)
    len_cumsum = np.cumsum(length)
    arange_orig = np.arange(len_cumsum[-1])
    len_cumsum_off = np.zeros(len_cumsum.size, dtype=len_cumsum.dtype)
    len_cumsum_off[1:] = len_cumsum[:-1]
    offset = np.repeat(len_cumsum_off, length)
    return arange_orig - offset

@timeit
def slice_idx(idx_orig, mask):
    return tuple(i[mask] for i in idx_orig)
    
def slice_pkt(pkt_orig : dict, mask : Union[tuple, list, np.ndarray]):
    return {k: v[mask] for k, v in pkt_orig.items()}

def merge_idx(*idxs):
    """
    Utility function to merge multiple sets of indices into one.
    Inputs
        each idxs[i] should be a n-tuple indexing into a nD array. 
    Output
        a single n-tuple
    Example:
        idxs = [(np.array([0,0]), np.array([0,1])), (np.array([1,1]), np.array([0,1]))]
        output = (np.array([0,0,1,1]), np.array([0,1,0,1]))
    """
    assert np.all(np.array([len(i) for i in idxs[1:]]) == len(idxs[0]))
    return tuple(np.concatenate(ii) for ii in zip(*idxs))

def merge_dict(*dicts):
    ret_temp = defaultdict(list)
    for dt in dicts:
        if dt is None:
            continue
        for k, v in dt.items():
            ret_temp[k].append(v)
    if len(ret_temp) == 0:
        return None
    else:
        return {k: np.concatenate(v, axis=0) for k, v in ret_temp.items()}




def check_intersect_allpairs(pos_allpairs, obs):

    assert len(pos_allpairs.shape) == 5 and len(obs) == 4
    pos1 = pos_allpairs[:, :, :, 0, :]
    pos2 = pos_allpairs[:, :, :, 1, :]
    if hasattr(obs[0], "__len__"):
        _cond11 = pos1[:, :, :, 0] >= obs[0][:, np.newaxis, np.newaxis]
        _cond12 = pos1[:, :, :, 1] >= obs[1][:, np.newaxis, np.newaxis]
        _cond13 = pos1[:, :, :, 0] <= obs[2][:, np.newaxis, np.newaxis]
        _cond14 = pos1[:, :, :, 1] <= obs[3][:, np.newaxis, np.newaxis]
        _cond21 = pos2[:, :, :, 0] >= obs[0][:, np.newaxis, np.newaxis]
        _cond22 = pos2[:, :, :, 1] >= obs[1][:, np.newaxis, np.newaxis]
        _cond23 = pos2[:, :, :, 0] <= obs[2][:, np.newaxis, np.newaxis]
        _cond24 = pos2[:, :, :, 1] <= obs[3][:, np.newaxis, np.newaxis]
    
    else:
        _cond11 = pos1[:, :, :, 0] >= obs[0]
        _cond12 = pos1[:, :, :, 1] >= obs[1]
        _cond13 = pos1[:, :, :, 0] <= obs[2]
        _cond14 = pos1[:, :, :, 1] <= obs[3]
        _cond21 = pos2[:, :, :, 0] >= obs[0]
        _cond22 = pos2[:, :, :, 1] >= obs[1]
        _cond23 = pos2[:, :, :, 0] <= obs[2]
        _cond24 = pos2[:, :, :, 1] <= obs[3]
    status_within = (_cond11 * _cond12 * _cond13 * _cond14) \
                  + (_cond21 * _cond22 * _cond23 * _cond24)
    def cross(p1, p2, p3, obstacle_B=False): 

        if obstacle_B:
            f_take = lambda p, i: p[i][:, np.newaxis, np.newaxis] if type(p) != np.ndarray else p[:, :, :, i]
        else:
            f_take = lambda p, i: p[i] if type(p) != np.ndarray else p[:, :, :, i]

        x1 = f_take(p2, 0) - f_take(p1, 0)
        y1 = f_take(p2, 1) - f_take(p1, 1)
        x2 = f_take(p3, 0) - f_take(p1, 0)
        y2 = f_take(p3, 1) - f_take(p1, 1)


        return x1*y2 - x2*y1 
    
    def segment(pvec1, pvec2, p3, p4):
        obstacle_B = hasattr(p3[0], "__len__")
        if obstacle_B:
            p30_p40 = np.concatenate((p3[0][:, np.newaxis], p4[0][:, np.newaxis]), axis=-1)
            p31_p41 = np.concatenate((p3[1][:, np.newaxis], p4[1][:, np.newaxis]), axis=-1)
            _cond1 = np.maximum(pvec1[:, :, :, 0], pvec2[:, :, :, 0]) >= np.min(p30_p40, axis=-1)[:, np.newaxis, np.newaxis]
            _cond3 = np.maximum(pvec1[:, :, :, 1], pvec2[:, :, :, 1]) >= np.min(p31_p41, axis=-1)[:, np.newaxis, np.newaxis]
            _cond2 = np.max(p30_p40, axis=-1)[:, np.newaxis, np.newaxis] >= np.minimum(pvec1[:, :, :, 0], pvec2[:, :, :, 0])
            _cond4 = np.max(p31_p41, axis=-1)[:, np.newaxis, np.newaxis] >= np.minimum(pvec1[:, :, :, 1], pvec2[:, :, :, 1])           
        else:
            _cond1 = np.maximum(pvec1[:, :, :, 0], pvec2[:, :, :, 0]) >= min(p3[0], p4[0])
            _cond3 = np.maximum(pvec1[:, :, :, 1], pvec2[:, :, :, 1]) >= min(p3[1], p4[1])
            _cond2 = max(p3[0], p4[0]) >= np.minimum(pvec1[:, :, :, 0], pvec2[:, :, :, 0])
            _cond4 = max(p3[1], p4[1]) >= np.minimum(pvec1[:, :, :, 1], pvec2[:, :, :, 1])
        _conda = cross(pvec1, pvec2, p3, obstacle_B) * cross(pvec1, pvec2, p4, obstacle_B) <= 0
        _condb = cross(p3, p4, pvec1, obstacle_B) * cross(p3, p4, pvec2, obstacle_B) <= 0
        return _cond1 * _cond2 * _cond3 * _cond4 * _conda * _condb
    
    p1_obs = [obs[0], obs[1]]
    p2_obs = [obs[2], obs[3]]
    p3_obs = [obs[2], obs[1]]
    p4_obs = [obs[0], obs[3]]
    status_cross = segment(pos1, pos2, p1_obs, p2_obs) + segment(pos1, pos2, p3_obs, p4_obs)

    return status_within + status_cross


def check_intersect(l1,l2,sq):

    def cross(p1,p2,p3): 
        x1=p2[0]-p1[0]
        y1=p2[1]-p1[1]
        x2=p3[0]-p1[0]
        y2=p3[1]-p1[1]
        return x1*y2-x2*y1     

    def segment(p1,p2,p3,p4):

        if (max(p1[0],p2[0])>=min(p3[0],p4[0]) 
            and max(p3[0],p4[0])>=min(p1[0],p2[0])
            and max(p1[1],p2[1])>=min(p3[1],p4[1]) 
            and max(p3[1],p4[1])>=min(p1[1],p2[1])
            and cross(p1,p2,p3)*cross(p1,p2,p4)<=0  
            and cross(p3,p4,p1)*cross(p3,p4,p2)<=0):
            D=1
        else:
            D=0
        return D


    if ((l1[0] >= sq[0] and l1[1] >= sq[1] and  l1[0] <= sq[2] and  l1[1] <= sq[3]) or 
        (l2[0] >= sq[0] and l2[1] >= sq[1] and  l2[0] <= sq[2] and  l2[1] <= sq[3])):
        return 1
    else:
        p1 = [sq[0],sq[1]]
        p2 = [sq[2],sq[3]]
        p3 = [sq[2],sq[1]]
        p4 = [sq[0],sq[3]]
        if segment(l1,l2,p1,p2) or segment(l1,l2,p3,p4):
            return 1
        else:
            return 0


def dBmSum_reduceat(rss_power_stacked, rss_max_power_stacked, len_per_batch_cumsum, noise):

    rss_sum = np.add.reduceat(rss_power_stacked, len_per_batch_cumsum)\
             - rss_max_power_stacked + np.power(10, noise/10)
    return 10 * np.log10(rss_sum)


def dBmSum(l_dBm):

    return 10 * math.log10(sum([10**(i / 10.0) for i in l_dBm]))

