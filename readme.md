# LAUREL: Learn prActical commUnication strategies in cooperative Multi-Agent REinforcement Learning
This repository is the official implementaton of [Learning Practical Communication Strategies in Cooperative Multi-Agent Reinforcement Learning](https://anrg.usc.edu/www/papers/hu22.pdf).

## Dependencies

* python==3.10.4
* pytorch==1.11.0
* numpy==1.22.3
* sacred==0.8.2 
* pyyaml==6.0



## Train

<!--- python marl/main.py --configs config_train/<env>/<yml> --gpu <idx> --->


```
python src/main.py --config=<alorithm> --env-config=<game env config> --rl-config=<algorithm config> --cuda=<gpu idx>
```

* `<algorithm>`: name of a yaml file stored in src/config/algs/`<algorithm>`.py. The file contains training parameters (e.g. action selector) for the specific algorithm. Use `qmix` for all off-policy based experiments in the paper.

* `<game env config>`: name of a yaml file stored in src/config/envs/`<game env config>`.yaml. The file contains game environment configurations (`env_args`), message encoder architecture (`embedder_type`) filename of wireless envrionment configuration (`constant_file`), wireless observation and message if existent (`msg_type`), etc. These settings need be consitant (e.g. TarMac should not use LAUREL's encoder).
 
* `<algorithm config>`: name of a yaml file stored in src/config/`<algorithm config>`.yaml. The file contains default training algorithm hyperparameters, etc. You do not need to modify anything in the file but `training agent` (use `rnn_agent` for LAUREL).



## Example

We demonstrate how to train the off-policy LAUREL in the Lumberjacks environment. To reproduce Figure 5 in the paper with 5 agents and 3 trees in a 10x10 field, we provide the corresponding yaml files specifying the game and wireless environment, and trainig hyperparameters.


To train off-policy LAUREL, 

<!--- LAUREL.yaml is the same as default.yaml. I believe users need not touch the file --->
```
python src/main.py --config=qmix --env-config=lj_LAUREL --rl-config=default --cuda=1
```

## Logs
You specify the training log directory in `CONFIG_TEMPLATE.yml`: `logging`-->`dir` --> `local`. By default, training logs including configuration yamls and evaluation metrics will be stored under the directory of `logs/<game>-<wifi>/<finished|running|crashed|killed>/`

Inference logs are under the directory of `logs/<game>-<wifi>/INF/`

In each inference subdirectory, I also created a symbol-link to the original subdirectory for the training run (Note that inference is always reloading a checkpoint generated from a previous training run).



## Code Structure


### Architecture
`src/modules/rnn_agent.py`: LAUREL architecture. 

`src/modules/mixer/qmix.py`: Q value "mixer"s of LAUREL.

`src/modules/gnn_embedder.py`: encoders of LAUREL (gin)

### Training and evaluation

`src/run.py`: wrapper for trainer, specifies the outer-most training loop for multiple episodes

`src/runners/episode_runner.py`: defines the training algorithm within a single episode /trajectory (including reply buffer, agent-environment interactions, etc.). 
### Environment

**Wrapper** `src/envs/wrapper.py` wrapping up the game environment and wireless environment.

**Game environment**
 `src/envs/games/lumberjack.py` for lumberjacks and `src/envs/games/predator_prey_vec.py` for vectorized version of predator prey. We implement configurable obstacles in these environments that will cauze signal attenuation and block agents' navigation. The vectorized version of predator prey can accelerate training for on-policy algorithms.



**Wireless environment** We implement a wireless envrionment where
(1) each agent senses the channel, contends the medium (with pCSMA) to send messages if its communication action;
(2) signals propagate in the environment with path loss, background noise, attenuation, and interferecne.
(3) agents senses channel and can successfully receive messages whose SINR is above the threshold
Note the wireless environment is parallel supporting batch rollout, which can accelerate training of on-policy algorithms.  

* `envs.py`: `EnvWirelessVector`
* `agents.py`: `Agents`
* `protocols.py`: `Protocols`
* `pcsma.py`: `pCSMAs(Protocols)`
* `data_struct.py`: `NetworkMeasure`, `Buffer`, `QueueHeader`
*`utils.py`
* `constants.py`: this may be moved out to some external `*.yml` file in the future. 

The dependencies among classes: 

```
                        GymWrapper
             _______________|_____________
            /                             \
PredatorPreyEnvWirelessVec         EnvWirelessVector
            \_____________________________/
                            |
                          Agents
                            |
                    pCSMAs(Protocols)
                            |
            NetworkMeasure / Buffer / QueueHeader
                     
```            
            
## Reference
We refered to pymarl implementation of QMIX algorithm (no wireless environment and no communication): https://github.com/oxwhirl/pymarl.

## License
Code licensed under the MIT License

## Citation

If you found the implementation or paper helpful, you are encouraged to cite our paper

```
@inproceedings{h22,
  title={Learning Practical Communication Strategies in Cooperative Multi-Agent Reinforcement Learning},
  author={Hu, Diyi and Zhang, Chi and Prasanna, Viktor and Krishnamachari, Bhaskar},
  booktitle={Asian Conference on Machine Learning},
  pages={},
  year={2022},
  organization={PMLR}
}

```