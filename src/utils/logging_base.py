import os
import torch
from dataclasses import dataclass, field, fields, InitVar
from typing import List, get_type_hints, Union
import re
import builtins
import numpy as np
import numbers
import shutil
from collections import defaultdict
import glob
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt

import pdb



_bcolors = {'header'    : '\033[95m',
            'blue'      : '\033[94m',
            'green'     : '\033[92m',
            'yellow'    : '\033[93m',
            'red'       : '\033[91m',
            'bold'      : '\033[1m',
            'underline' : '\033[4m',
            ''          : '\033[0m',
            None        : '\033[0m'}

def l_div(l1, l2, idx_start, idx_end):
    if len(l1) == 0:
        return l1
    assert isinstance(l2, numbers.Number) or len(l1) == len(l2)
    if idx_start is None:
        idx_start = 0
    if idx_end is None:
        idx_end = len(l1)
    if isinstance(l2, numbers.Number):
        return [v / l2 for i, v in enumerate(l1[idx_start:idx_end])]
    else:
        return [v / l2[i + idx_start] for i, v in enumerate(l1[idx_start:idx_end])]
    

@dataclass
class MetricsCommon:
    test_return         : List[np.ndarray] = field(default_factory=list)
    test_num_steps      : List[int]   = field(default_factory=list)
    test_comm_action_avg: List[int]   = field(default_factory=list)
    td_error_abs        : List[float] = field(default_factory=list)
    loss                : List[float] = field(default_factory=list)
    grad_norm           : List[float] = field(default_factory=list)
    q_taken             : List[float] = field(default_factory=list)
    epsilon             : List[float] = field(default_factory=list)
    reward              : List[np.ndarray] = field(default_factory=list)        
    success             : List[float] = field(default_factory=list)
    comm_action         : List[float] = field(default_factory=list)
    value_loss          : List[float] = field(default_factory=list)
    action_loss         : List[float] = field(default_factory=list)
    entropy             : List[float] = field(default_factory=list)
    num_episodes        : List[int]   = field(default_factory=list)
    num_steps           : List[int]   = field(default_factory=list)
    time_epoch          : List[float] = field(default_factory=list)
    broadcast_suc_rate  : List[float] = field(default_factory=list)
    msg_0_recv_rate     : List[float] = field(default_factory=list)
    msg_1_recv_rate     : List[float] = field(default_factory=list)
    comm_position       : List[float] = field(default_factory=list)            
    comm_step           : List[float] = field(default_factory=list)            
    predator_init_pos   : List[float] = field(default_factory=list)             

    best_model_metric = None  
    best_epoch = None
    _num_epochs_prev = None

    normalized      : InitVar[bool] = False
    def __post_init__(self, normalized):
        self.norm_group_ep = {'reward', 'success', 'num_steps'}
        self.norm_group_st = {'comm_action', 'value_loss', 'action_loss', 'entropy'}
        if not normalized:
            self.num_steps, self.num_episodes = [], []

    def extract(self, idx_start, idx_end, normalize):
        if normalize:
            args = {m: l_div(getattr(self, m), self.num_episodes, idx_start, idx_end) for m in self.norm_group_ep}
            args.update({m: l_div(getattr(self, m), self.num_steps, idx_start, idx_end) for m in self.norm_group_st})
            args.update({m.name: l_div(getattr(self, m.name), 1, idx_start, idx_end) for m in fields(self) if m.name not in self.norm_group_ep and m.name not in self.norm_group_ep})
        else:
            args = {m.name: getattr(self, m.name)[idx_start:idx_end] for m in fields(self)}
        return self.__class__(**args, normalized=True)

    def is_best_model(self, epoch):
        self._num_epochs_prev = max(len(self.reward), len(self.test_return))
        self.best_epoch = self._num_epochs_prev - 1
        assert epoch == self._num_epochs_prev - 1
        return True

class MetricsPPSteps(MetricsCommon):
    def is_best_model(self, epoch):
        model_to_update = False
        if self.best_model_metric is None:
            assert self.best_epoch is None or self.best_epoch < 0
            self.best_epoch = np.argmin(self.num_steps)
            self.best_model_metric = self.num_steps[self.best_epoch]
            model_to_update = True
        else:
            ep_offset = np.argmin(self.num_steps[self._num_epochs_prev:])
            epoch_best_candy = self._num_epochs_prev + ep_offset
            num_steps_candy = self.num_steps[epoch_best_candy]
            if num_steps_candy < self.best_model_metric:
                self.best_model_metric = num_steps_candy
                self.best_epoch = epoch_best_candy
                model_to_update = True
            else:
                model_to_update = False
        self._num_epochs_prev = len(self.num_steps)
        assert epoch == self._num_epochs_prev - 1
        return model_to_update

class LoggerBase:
    def __init__(
        self, 
        config_dict, 
        dir_log, 
        dir_inference,
        timestamp, 
        git_rev, 
        MetricsCls=MetricsCommon,
        no_log=False,
        **kwargs
    ):
        """
        Inputs:
            metrics     list of metrics to log
        """
        self.task = 'inference' if dir_inference is not None else 'train'
        self.timestamp = timestamp
        _dir_log_key = '_' if config_dict['logger_key'] else ''
        subdir_task = 'INF' if self.task == 'inference' else ''
        subdir_stat = '' if self.task == 'inference' else 'running'
        self.dir_log = (f"{dir_log}/{config_dict['env']['game']['name']}-{config_dict['env']['wifi']['name']}{_dir_log_key}{config_dict['logger_key']}"
                        f"/{subdir_task}/{subdir_stat}/{timestamp}-{git_rev.strip():s}/")
        self.metrics = MetricsCls()

        self.config_dict = config_dict
        self.file_ep = {"train" : f"{self.dir_log}/epoch_train.log",
                        "val"   : f"{self.dir_log}/epoch_val.log",
                        "test"  : f"{self.dir_log}/epoch_test.log"}
        self.file_final = f"{self.dir_log}/final.log"
        self.is_header_written = {"train": False, "val": False, "test": False, "final": False}
        self.epoch_current = -1
        self.epoch_last_logged = -1
        if not os.path.exists(self.dir_log):
            os.makedirs(self.dir_log)

        if self.task == 'inference':
            self.dir_inference = dir_inference
        else:
            self.path_saver = f"{self.dir_log}/saved_model_{self.timestamp}.pkl"
            self.dir_saver = f'{self.dir_log}/'
        f_time_prof_init = lambda : [[]] 
        self.time_profile = {'env_game': defaultdict(f_time_prof_init), 
                             'env_wifi': defaultdict(f_time_prof_init), 
                             'marl'    : defaultdict(f_time_prof_init)}
        self.name_fields_log, self.value_fields_log = None, None       
        self.no_log = no_log

    def _metric_name_conversion(self, input_name):
        """
        Convert the LAUREL metric name to our own metric name
        """
        if input_name.endswith('_std'):
            return None
        if input_name.endswith('_mean'):
            input_name = input_name[:-5]
        alias = {'test_return': 'test_return',
                 'test_ep_length': 'test_num_steps',
                 'td_error_abs': 'td_error_abs',
                 'loss': 'loss',
                 'grad_norm': 'grad_norm',
                 'q_taken': 'q_taken',
                 'epsilon': 'epsilon',
                 'test_comm_action_avg': 'test_comm_action_avg'}
        if input_name in alias:
            input_name = alias[input_name]
        else:
            input_name = None
        return input_name

    def add_metrics(self, kv_field : dict, epoch : int):
        assert epoch == self.epoch_current
        for k, v in kv_field.items():
            if type(v) == torch.Tensor and len(v.shape) == 0:
                v = v.data.item()
            getattr(self.metrics, k).append(v)
    
    @staticmethod
    def stringf(msg : Union[str, dict], style='', ending=''):

        style = '' if not style else style
        if type(msg) == str:
            subs = msg
        elif type(msg) == dict:
            _default_separator = '\t'
            subs = ''.join((f"{_bcolors[se if type(se) == str or se is None else se[0]]}"
                            f"{m}"
                            f"{_default_separator if type(se) == str or se is None else se[1]}") 
                                for m, se in msg.items())
            subs = subs.strip()
        else:
            raise NotImplementedError
        return f"{_bcolors[style]}{subs}{_bcolors[None]}{ending}"

    @staticmethod
    def printf(msg, style=''):
        print(LoggerBase.stringf(msg, style=style, ending=''))

    @staticmethod
    def add_logger_args(parser):
        pass

    @staticmethod
    def _write2file(filename, logstr, write_mode="a"):
        with open(filename, write_mode) as f:
            f.write(logstr)        
    
    def save_model(self, learner, epoch : int):

        assert epoch == self.epoch_current
        self.printf(f"  [E{epoch}] Saving model ...", style="yellow")
        learner.save_models(self.dir_saver)
        
    def restore_model(self, model):
 
        if self.metrics.best_epoch >= 0:
            model.load_state_dict(torch.load(self.path_saver))
            self.printf("  Restoring model ...")
        else:
            self.printf("  NOT restoring model ... PLS CHECK!")
    
    def load_model(self, model_dict):

        assert hasattr(self, 'dir_inference') and self.dir_inference is not None
        fname_pkl = glob.glob(f"{self.dir_inference}/*.pkl")
        assert len(fname_pkl) == 1
        fname_pkl = fname_pkl[0]
        model_dict_to_load = torch.load(fname_pkl)
        for skip in ['mask_self', 'agent_mask']:
            assert not model_dict_to_load['policy_net'][skip].requires_grad
            model_dict_to_load['policy_net'][skip] = model_dict['policy_net'].state_dict()[skip]
        for k, v in model_dict_to_load.items():
            model_dict[k].load_state_dict(v)

    
    def record_timing(self, time_profile):
        for k, v in time_profile.items():
            for kk, vv in v.items():
                self.time_profile[k][kk][-1].append(vv)

    def update_epoch(self, ep, silent=True):
        self.epoch_current = ep
        if not silent:
            self.printf(f"Epoch {ep:4d}")
        for k, v in self.time_profile.items():
            for kk in v:
                self.time_profile[k][kk].append([])

    def update_best_model(self, ep, model_state : dict):
        if self.metrics.is_best_model(ep):
            self.save_model(model_state, ep)

    def _aggr_epoch_stat(self, metrics, name, convert_to_str=True):
        assert (name not in ['comm_position', 'comm_step', 'predator_init_pos'])
        val = getattr(metrics, name)
        if len(val) == 0:
            return ['-']
        else:
            aggr = sum(val) / len(val)
            if type(aggr) in [np.ndarray, list]:
                return aggr if not convert_to_str else [self._numerical_format(a) for a in aggr]
            else:
                return [aggr] if not convert_to_str else [self._numerical_format(aggr)]


    def _numerical_format(self, val):
        if type(val) == float or type(val) == np.float64 or type(val) == np.float32:
            return f"{val:.4f}"
        elif type(val) == int:
            return f"{val:6d}"
        else:
            print(f"GET NUMERICAL VALUE OF TYPE {type(val)}")
            raise NotImplementedError


    def log2file2screen(self, mode, epoch, metrics_onscreen=None, normalize=False):

        assert mode in ["train", "val", "test", "final"]
        assert epoch == self.epoch_current
        metrics_norm = self.metrics.extract(self.epoch_last_logged + 1, self.epoch_current + 1, normalize)
        msg = ""
        if not self.is_header_written[mode]:
            header = ["epoch"]
            for fm in fields(metrics_norm):
                val = getattr(metrics_norm, fm.name)
                if type(val) == list and len(val) > 0:
                    if isinstance(val[-1], numbers.Number):
                        header.append(f"{fm.name}")
                    elif fm.name in ['comm_position', 'comm_step', 'predator_init_pos']:
                        pass
                    else:       
                        header.extend(f"{fm.name}{i}" for i in range(len(val[-1])))
                else:
                    header.append(fm.name)
            self.name_fields_log = header
            msg += ", ".join(header) + "\n"
            self.is_header_written[mode] = True
        _msg_nested = [self._aggr_epoch_stat(metrics_norm, fm.name, convert_to_str=True) for fm in fields(metrics_norm) if fm.name not in ['comm_position', 'comm_step', 'predator_init_pos']]
        _msg_nested_num = [self._aggr_epoch_stat(metrics_norm, fm.name, convert_to_str=False) for fm in fields(metrics_norm) if fm.name not in ['comm_position', 'comm_step', 'predator_init_pos']]
        self.value_fields_log = [epoch] + [i for mn in _msg_nested_num for i in mn]

        if mode=='test':
            for fm_name in ['comm_position', 'comm_step', 'predator_init_pos']:
                val = getattr(metrics_norm, fm_name)
                if len(val) == 0: 
                    continue
                kk, vv = list(val[-1].keys()), list(val[-1].values()) 
                if fm_name == 'comm_position':
                    dim = self.config_dict['env']['game']['size']
                    comm_position = np.zeros((dim, dim))
                    comm_position[[kkk[0] for kkk in kk], [kkk[1] for kkk in kk]] = vv
 
                    sns.heatmap(comm_position, annot=True).get_figure().savefig('comm_position.png')
                    plt.clf()
                if fm_name == 'comm_step':
                    max_step = self.config_dict['params']['max_steps']
                    comm_step = np.zeros(max_step)
                    comm_step[kk] = vv
                    g = sns.barplot(x=np.arange(max_step), y=comm_step)
                    for cs_idx, cs_v in enumerate(comm_step):
                        g.text(cs_idx, int(cs_v), int(cs_v), color='black', ha="center")
                    g.set(ylim=(0, comm_step.max()+50))
                    g.get_figure().savefig('comm_step.png')
                    plt.clf()
                if fm_name == 'predator_init_pos':
                    dim = self.config_dict['env']['game']['size']
                    predator_init_pos = np.zeros((dim, dim))
                    predator_init_pos[[kkk[0] for kkk in kk], [kkk[1] for kkk in kk]] = vv

                    sns.heatmap(predator_init_pos, annot=True).get_figure().savefig('predator_init_pos.png')
                    plt.clf()     



        _msg_nested_dict = {fm.name: self._aggr_epoch_stat(metrics_norm, fm.name) for fm in fields(metrics_norm) if fm.name not in ['comm_position', 'comm_step', 'predator_init_pos']}
        msg += ", ".join([str(epoch)] + [i for mn in _msg_nested for i in mn]) + "\n"
        self._write2file(self.file_ep[mode], msg)
        self.epoch_last_logged = self.epoch_current

        if metrics_onscreen is not None:
            assert type(metrics_onscreen) == dict
            s_info = {f"Epoch {self.epoch_current:5d}": None}
            s_info.update({f"{n} = {', '.join(_msg_nested_dict[n])}": s for n, s in metrics_onscreen.items()})
            self.printf(s_info)

    @staticmethod
    def HELPER_cleanup_log_files(log_dir : str, cond_filter_logs : dict, be_cautious : bool=True):

        assert os.path.isdir(log_dir), "input must be a valid logging directory"
        if be_cautious:

            dir_chd = os.path.realpath(log_dir)
            dir_pnt = os.path.realpath(".")
            pnt2chd = os.path.relpath(dir_chd, start=dir_pnt)
            assert not pnt2chd.startswith(os.pardir)
            name_contains_log = False
            for d in log_dir.split('/'):
                if "log" in d.split("_") or "log" in d.split(" ") \
                        or "logs" in d.split("_") or "logs" in d.split(" ") \
                        or "Log" in d.split("_") or "Log" in d.split(" ") \
                        or "Logs" in d.split("_") or "Logs" in d.split(" "):
                    name_contains_log = True
                    break
            assert name_contains_log, "BE CAUTIOUS! the input log_dir does not contain 'log' in its directory name"
 
            subdir2remove = []
            for d in os.listdir(log_dir):

                if os.path.isfile(os.path.join(log_dir, d)):
                    continue
                if len(os.listdir(os.path.join(log_dir, d))) == 0:
                    subdir2remove.append(d)
                    continue
                for f in os.listdir(os.path.join(log_dir, d)):
                    assert os.path.isfile(os.path.join(log_dir, d, f)), "Log dir contains subdirectories. I don't understand this format"
                    for f_re, cond in cond_filter_logs.items():
                        if not re.fullmatch(f_re, f):
                            continue
                        with open(os.path.join(log_dir, d, f), 'r') as fread:
                            if len(fread.readlines()) < cond:
                                subdir2remove.append(d)
                                break
            if be_cautious:
                s_head = f"REMOVING {len(subdir2remove):5d} LOGGING DIRECTORIES"
                LoggerBase.printf(s_head, style="red")
                LoggerBase.printf("=" * len(s_head), style="red")
                for d in subdir2remove:
                    LoggerBase.printf(d)
                LoggerBase.printf("")
                LoggerBase.printf({"Confirm remove? This operation cannot be undone.": "yellow", "[Y/n]": "red"})
                choice = input()
                if choice != "Y":
                    LoggerBase.printf("Operation cancelled by user. Bye!", style="red")
                    return
            for d in subdir2remove:
                shutil.rmtree(os.path.join(log_dir, d))
            LoggerBase.printf("Operation finished. Bye!", style="yellow")

    @staticmethod
    def DEBUG_print_model_key_stat(model):

        for k, v in model.state_dict().items():
            LoggerBase.printf({f"{k}": ("yellow", " "),
                               f"shape = {v.shape}": (None, "\t\n\t"), 
                               f"min = {v.min():.4f}": None, 
                               f"max = {v.max():.4f}": None,
                               f"mean = {v.mean():.4f}": None,
                               f"std = {v.std():.4f}": None})

    def print_time_profile(self):

        time_component = {k1: {k2: None for k2 in v1} for k1, v1 in self.time_profile.items()}
        self.printf("AVERAGE PER EPOCH TIME BREAKDOWN:", style='bold')
        for k, v in self.time_profile.items():
            for kk, vv in v.items():
                assert len(vv) == self.epoch_current + 1
                time_component[k][kk] = sum(sum(vi) for vi in vv) / len(vv)
        total_time = sum(v2 for k1, v1 in time_component.items() for k2, v2 in v1.items())
        for k, v in self.time_profile.items():
            self.printf(k, style='bold')
            for kk, vv in v.items():
                time_c = time_component[k][kk]
                self.printf(f'{kk:>45s}: {time_c:<.2f}s ({time_c / total_time * 100:5.2f}%)')
        self.printf(f"Time for all profiled components: {total_time:.3f}s (100%)")
    
    def cpy_cfg_yml(self, path_yml):
        shutil.copyfile(path_yml, f"{self.dir_log}/{path_yml.split('/')[-1]}")
        if self.task == 'inference':
            src_sym = os.path.relpath(self.dir_inference, self.dir_log)
            os.symlink(src_sym, f"{self.dir_log}/orig_train_log")


    def end_training(self, status):
        assert status in ['crashed', 'finished', 'killed']
        if self.no_log:
            from itertools import product
            if os.path.exists(self.dir_log):
                assert os.path.isdir(self.dir_log)
                f_ymlpt = os.listdir(self.dir_log)
                if len(f_ymlpt) == 1:
                    assert f_ymlpt[0].split('.')[-1] in ['yml', 'yaml'], \
                        f"DIR {self.dir_log} CONTAINS UNKNOWN TYPE OF FILE. ABORTING!"
                else:
                    assert len(f_ymlpt) == 2
                    assert all(os.path.isfile(f"{self.dir_log}/{f}") for f in f_ymlpt)
                    ext1 = ['yml', 'yaml']
                    ext2 = ['pkl', 'pt']
                    assert set(f.split('.')[-1] for f in f_ymlpt) in [set(p) for p in product(ext1, ext2)], \
                        f"DIR {self.dir_log} CONTAINS UNKNOWN TYPE OF FILE. ABORTING!"
                shutil.rmtree(self.dir_log)
                self.printf(f"Successfully removed log dir {self.dir_log}!", style='red')
        else:
            dir_split = self.dir_log.split('/')     
            dir_split = dir_split if dir_split[-1] != '' else dir_split[:-1]
            assert dir_split[-2] == 'running'
            dir_split[-2] = status
            dir_new_parent = '/'.join(dir_split[:-1])
            if not os.path.exists(dir_new_parent):
                os.makedirs(dir_new_parent)
            assert os.path.isdir(self.dir_log) and os.path.isdir(dir_new_parent)
            f_logfiles = os.listdir(self.dir_log)
            assert all(os.path.isfile(f"{self.dir_log}/{f}") for f in f_logfiles)
            shutil.move(self.dir_log, dir_new_parent)
            self.printf(f"Successfully moved {self.dir_log} to {dir_new_parent}", style='red')