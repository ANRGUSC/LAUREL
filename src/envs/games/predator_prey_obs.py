"""
This script generates random walls (obstacles) in the
PP grid environment. 
"""
import numpy as np

def generate_one_wall(batch_size, dim, size_obs, not_touch_edge=False):

    idx_batch_obs_horizon = np.random.choice(batch_size, np.int(np.floor(batch_size/2)), replace=False)
    mask_batch_obs_horizon = np.zeros(batch_size, dtype=bool)
    mask_batch_obs_horizon[idx_batch_obs_horizon] = True
    obs_start_pos_free = (np.random.randint(dim - 2, size=batch_size) + 1)[:, np.newaxis]  
    
    if not_touch_edge:
        obs_start_pos_constr = np.random.choice(np.arange(1, (dim - size_obs)), size=batch_size)[:, np.newaxis]
    else:
        obs_start_pos_constr = np.random.randint((dim - size_obs + 1), size=batch_size)[:, np.newaxis]

    obs_pos_free = np.repeat(obs_start_pos_free, size_obs, axis=-1)
    obs_pos_constr = obs_start_pos_constr + np.arange(size_obs)[np.newaxis,:]

    obs_pos = np.zeros((batch_size, size_obs, 2))
    obs_pos[mask_batch_obs_horizon] = np.concatenate((obs_pos_free[mask_batch_obs_horizon][:, :, np.newaxis], obs_pos_constr[mask_batch_obs_horizon][:, :, np.newaxis]), axis=-1)   # (x, y) in PP coordinate
    obs_pos[~mask_batch_obs_horizon] = np.concatenate((obs_pos_constr[~mask_batch_obs_horizon][:, :, np.newaxis], obs_pos_free[~mask_batch_obs_horizon][:, :, np.newaxis]), axis=-1)   # (x, y) in PP coordinate
    return obs_pos[:, np.newaxis, :, :], mask_batch_obs_horizon[:, np.newaxis]


def merge_obs(pos1, mask_horiz1, pos2, mask_horiz2, dim):
    """
    Set the overlapped part in-place in the ndarray
    """
    pos1_raw = pos1[:, 0, :, 0]  * dim + pos1[:, 0, :, 1]
    pos2_raw = pos2[:, 0, :, 0]  * dim + pos2[:, 0, :, 1]
    obstacle_size = pos1_raw.shape[-1]
    num_grids_overlap = ((np.tile(pos1_raw, obstacle_size)-np.repeat(pos2_raw, obstacle_size, axis=-1))==0).sum(axis=-1)

    idx_batch_obs_overlap = np.where(num_grids_overlap > 1)[0]
    if len(idx_batch_obs_overlap):
        
        x_obs_pos_hrztl = pos1[idx_batch_obs_overlap, 0, 0, 0][mask_horiz1[idx_batch_obs_overlap, 0]]
        mask_shift_up = x_obs_pos_hrztl > 2
        idx_batch_shift_up = idx_batch_obs_overlap[mask_horiz1[idx_batch_obs_overlap, 0]][mask_shift_up]
        idx_batch_shift_up_mod = idx_batch_obs_overlap[mask_horiz1[idx_batch_obs_overlap, 0]][~mask_shift_up]
        pos1[idx_batch_shift_up, 0, :, 0] -= 2
        pos1[idx_batch_shift_up_mod, 0, :, 0] += dim - 4

        y_obs_pos_vtcl = pos1[idx_batch_obs_overlap, 0, 0, 1][~mask_horiz1[idx_batch_obs_overlap, 0]]
        mask_shift_left = y_obs_pos_vtcl > 2
        idx_batch_shift_left = idx_batch_obs_overlap[~mask_horiz1[idx_batch_obs_overlap, 0]][mask_shift_left]
        idx_batch_shift_left_mod = idx_batch_obs_overlap[~mask_horiz1[idx_batch_obs_overlap, 0]][~mask_shift_left]
        pos1[idx_batch_shift_left, 0, :, 1] -= 2
        pos1[idx_batch_shift_left_mod, 0, :, 1] += dim - 4

              
        pos1 = np.concatenate((pos1, pos2), axis=1)
        mask_horiz1 = np.concatenate((mask_horiz1, mask_horiz2), axis=1)


def generate_all_obstacles(config_obs, batch_size, dim):
    assert config_obs is not None
    size_obs = config_obs['size']
    if config_obs['random'] in ['random', 'adjacent']:
        assert config_obs['num'] in [1, 2], "we do not support more than 2 walls"
        obs_pos, mask_batch_obs_horizon = generate_one_wall(batch_size, dim, size_obs)
        if config_obs['num'] > 1:
            obs_pos_2, mask_batch_obs_horizon_2 = generate_one_wall(batch_size, dim, size_obs, not_touch_edge=True)
            merge_obs(obs_pos, mask_batch_obs_horizon, obs_pos_2, mask_batch_obs_horizon_2, dim)
    elif config_obs['random'] == 'fixed':
        obs_pos = np.repeat(config_obs['location'][np.newaxis, :, :, :], batch_size, axis=0)
        mask_batch_obs_horizon = np.repeat(config_obs['horizontal'][np.newaxis, :], batch_size, axis=0)
    else:
        raise NotImplementedError
    return obs_pos, mask_batch_obs_horizon

def generate_random_adj_prey(obs_pos, mask_obs_horizon):
    """
    Returns:
        prey position
        index of invalid prey       used for predator generation later on
    """
    idx_prey_invalid = np.arange(obs_pos.shape[0])
    prey_pos = np.zeros((obs_pos.shape[0], 2))
    for i in range(obs_pos.shape[1]):
        jshuffle = np.arange(obs_pos.shape[2])
        SFLPREY = True  
        if SFLPREY:
            rng = np.random.default_rng()
            jshuffle = (rng.permuted(np.repeat(jshuffle[np.newaxis, :], obs_pos.shape[0], axis=0), axis=1)).T
        for j in jshuffle:
            if len(idx_prey_invalid)==0:
                break
            if SFLPREY:
                prey_pos[idx_prey_invalid] = obs_pos[idx_prey_invalid, [i for _ in idx_prey_invalid], j[idx_prey_invalid], :].copy()
            else:
                prey_pos[idx_prey_invalid] = obs_pos[idx_prey_invalid, i, j, :].copy()
            prey_pos[idx_prey_invalid, 1] -= (1 - mask_obs_horizon[idx_prey_invalid, i])
            prey_pos[idx_prey_invalid, 0] -= mask_obs_horizon[idx_prey_invalid, i]
            
            idx_relative_prey_invalid = np.unique(np.where(((prey_pos[idx_prey_invalid, np.newaxis, np.newaxis, :] 
                                                                == obs_pos[idx_prey_invalid]).sum(axis=-1))==2)[0])
            idx_prey_invalid = idx_prey_invalid[idx_relative_prey_invalid]
    return prey_pos, idx_prey_invalid


def generate_random_predator_prey(obs_pos, batch_size, dim, predator_pos, prey_pos, idx_prey_remain):

    npredator, nprey = predator_pos.shape[1], prey_pos.shape[1]

    obs_pos_raw = (obs_pos[:, :, 0] * dim + obs_pos[:, :, 1]).astype(int)

    mask_not_occupied = np.full((batch_size, dim**2), True)
    idx_batch_obs = np.arange(batch_size)[:, np.newaxis]
    mask_not_occupied[idx_batch_obs, obs_pos_raw] = False

    num_not_occupied = mask_not_occupied.sum(axis=-1)
    unique_num_not_occupied = np.unique(num_not_occupied)
    num_samples = npredator + nprey
    for u in unique_num_not_occupied:
        idx_batch_u = np.where(num_not_occupied == u)[0]        
        idx_pos_sampled = np.repeat(np.arange(u)[np.newaxis, :], len(idx_batch_u), axis=0)
        rng = np.random.default_rng()
        idx_pos_sampled = rng.permuted(idx_pos_sampled, axis=1)
        idx_pos_sampled = idx_pos_sampled[:, :num_samples]
        idx_not_occupied = np.where(mask_not_occupied[idx_batch_u])[1].reshape(len(idx_batch_u), u)
        pos_raw_sampled = idx_not_occupied[np.repeat(np.arange(len(idx_batch_u)), num_samples), idx_pos_sampled.flatten()].reshape(len(idx_batch_u), num_samples)
        x_sampled, y_sampled = np.unravel_index(pos_raw_sampled, (dim, dim))
        pos_sampled = np.concatenate((x_sampled[:, :, np.newaxis], y_sampled[:, :, np.newaxis]), axis=-1)
        predator_pos[idx_batch_u] = pos_sampled[:, :npredator, :]

        idx_prey_random = idx_batch_u
        if idx_prey_remain is not None:
            idx_prey_random = np.intersect1d(idx_prey_remain, idx_batch_u, assume_unique=True)
        mask_prey_random = np.zeros(batch_size)
        mask_prey_random[idx_prey_random] = True 
        prey_pos[idx_prey_random] = pos_sampled[mask_prey_random[idx_batch_u].astype(bool)][:, npredator:, :]