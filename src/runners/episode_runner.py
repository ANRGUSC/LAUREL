from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1
        if hasattr(self.args, 'wifi_args'):
            self.env = env_REGISTRY[self.args.env](wifi_args=self.args.wifi_args, **self.args.env_args)
        else:
            self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -1000000
        self.action_per_step = []
        self.step_first_catch_prey = []

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False, render=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        self.viz.init_batch(1)
        first_step_catching = -1
        while not terminated:

            predator_pos_prev = self.env.env.agent.predator.pos.copy()
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
            self.batch.update(pre_transition_data, ts=self.t)

            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            hidden_states_copy = self.mac.hidden_states.detach().clone().cpu().numpy()
            if len(hidden_states_copy.shape) == 3:
                assert hidden_states_copy.shape[0] == 1
                hidden_states_copy = hidden_states_copy[0]
            if self.mac.attention_keys is not None:
                attention_keys_copy = self.mac.attention_keys.detach().clone().cpu().numpy()
                if len(attention_keys_copy.shape) == 3:
                    assert attention_keys_copy.shape[0] == 1
                    attention_keys_copy = attention_keys_copy[0]

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            if hasattr(self.env, 'with_comm'):
                comm_msg_data = {
                    'comm_mask': [self.env.recv_mask],
                    'comm_measure': [self.env.msg_measure]
                }
                if self.env.msg_replay == 'stored':
                    comm_msg_data['comm_msg'] = [self.env.augment_comm_msg(hidden_states_copy)] 
                    if self.mac.attention_keys is not None:
                        comm_msg_data['attn_msg'] = [attention_keys_copy]
            self.batch.update(comm_msg_data, ts=self.t + 1)

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)
            if self.env.env.name.startswith("PP"):
                agents_pos = {"predator": predator_pos_prev, "prey": self.env.env.agent.prey.pos.copy()}
            elif self.env.env.name.startswith("LJ"):
                agents_pos = {"predator": predator_pos_prev, "tree": self.env.env.agent.tree.pos.copy()}
            else:
                raise NotImplementedError
            self.viz.add_viz_state(
                {"position"  : agents_pos}, 
                self.env.action_comm, None if self.env.env.obs_pos is None else self.env.env.obs_pos, {0: 0}
            )            
            self.t += 1
            if self.env.env.agent.predator.mask_freeze.sum() > 1 and first_step_catching < 0:
                first_step_catching = self.t
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t
        elif self.env.env.name == 'PP':
            self.step_first_catch_prey.append(first_step_catching)
            self.action_per_step.append(self.env.action_comm_trajectory)
            cur_stats['comm_action_avg'] = self.env.avg_comm_action() + cur_stats.get('comm_action_avg', 0)
        elif self.env.env.name.startswith("LJ"):
            self.action_per_step.append(self.env.action_comm_trajectory)
            cur_stats['comm_action_avg'] = self.env.avg_comm_action() + cur_stats.get('comm_action_avg', 0)
        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix, test_mode=test_mode)
        elif (not test_mode) and (self.t_env - self.log_train_stats_t >= self.args.runner_log_interval):
            self._log(cur_returns, cur_stats, log_prefix, test_mode=test_mode)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        if render:
            self.viz.render(None, [-1])

        return self.batch

    def _log(self, returns, stats, prefix, test_mode=False):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env, test_mode=test_mode)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env, test_mode=test_mode)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env, test_mode=test_mode)
        stats.clear()
