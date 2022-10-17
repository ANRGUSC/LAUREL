import torch as th


def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):

    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
    for t in range(ret.shape[1] - 2, -1,  -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    return ret[:, 0:-1]


def get_gitrev_timestamp():
    import subprocess, time, datetime
    git_rev = subprocess.Popen(
        "git rev-parse --short HEAD",
        shell=True,
        stdout=subprocess.PIPE,
        universal_newlines=True,
    ).communicate()[0]
    timestamp = time.time()
    timestamp = datetime.datetime.fromtimestamp(int(timestamp)).strftime(
        "%Y-%m-%d %H-%M-%S"
    )
    return git_rev, timestamp


def setup_global_config(debug=False):

    import yaml
    from os import path
    import sys
    import importlib
    fname_global_config = 'CONFIG.yml'
    if not path.exists(fname_global_config):
        fname_global_config = 'CONFIG_TEMPLATE.yml'
        assert path.exists(fname_global_config)
        print(f"LOADING {fname_global_config}. PLS DOUBLE-CHECK!!")
    with open(fname_global_config) as f_config:
        meta_config = yaml.load(f_config, Loader=yaml.FullLoader)

    def my_import(name):
        components = name.split('.')
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod

    if debug:      
        ret_module = my_import('utils.logging_base.LoggerBase')
    else:
        cls_logger = meta_config['logging']['logger'].split('.')
        ret_module = my_import('.'.join(cls_logger[:-1]) + f'.{cls_logger[-1]}')
    return ret_module, meta_config