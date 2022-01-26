from collections import defaultdict
import logging
import numpy as np
import torch

class Logger:
    def __init__(self, console_logger, logger_customize=None):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False

        self.stats = defaultdict(lambda: [])

        # NOTE: logger_customize is only used for tracking test metrics
        self.logger_customize = logger_customize

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value
        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_stat(self, key, value, t, to_sacred=True, test_mode=False):
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]
        
        if self.logger_customize is not None:
            _ep = self.logger_customize.epoch_current
            _key = self.logger_customize._metric_name_conversion(key)
            if _key is not None:
                self.logger_customize.add_metrics({_key: value}, _ep)
    
    def update_test_epoch(self, silent=True):
        """
        Update the test epoch number for the logger_customize.
        We update "test" epoch since we now decide to not log training perf in customized logger. 
        """
        ep_new = self.logger_customize.epoch_current + 1
        self.logger_customize.update_epoch(ep_new, silent=silent)

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            my_list = [x[1] for x in self.stats[k][-window:]]
            item = f"{torch.mean(torch.stack(my_list), dim=0):.4f}" if k == "grad_norm" else f"{np.mean(my_list):.4f}"
            #item = "{:.4f}".format(np.mean([x[1] for x in self.stats[k][-window:]]))
            log_str += f"{k + ':':<25}{item:>8}"
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)

        if self.logger_customize is not None:
            _ep = self.logger_customize.epoch_current
            self.logger_customize.log2file2screen('test', _ep)


# set up a custom logger
def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')

    return logger

