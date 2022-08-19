# LNTransactionFee
Lightning Network Transaction Fee Solver


<div>

[![Made with Python](https://img.shields.io/badge/Python->=3.8-red?logo=python&logoColor=white)](https://python.org "Go to Python homepage")
[![pandas - up-to-date](https://img.shields.io/static/v1?label=pandas&message=up-to-date&color=blueviolet)](https://pandas.pydata.org/)
[![Gym - >=0.2](https://img.shields.io/static/v1?label=Gym&message=>%3D0.2&color=black)](https://github.com/openai/gym)
[![Stable-baselines3 - >= 1.6](https://img.shields.io/static/v1?label=Stable-baselines3&message=>%3D+1.6&color=2ea44f)](https://stable-baselines3.readthedocs.io/en/master/)
[![Sb3-contrib - >= 1.6](https://img.shields.io/static/v1?label=Sb3-contrib&message=>%3D+1.6&color=green)](https://sb3-contrib.readthedocs.io/en/master/)

[![NetworkX - ~2.8.5](https://img.shields.io/static/v1?label=NetworkX&message=~2.8.5&color=brightgreen)](https://networkx.org/)
[![tensorboard - ~=2.9.1](https://img.shields.io/static/v1?label=tensorboard&message=~%3D2.9.1&color=orange)](https://www.tensorflow.org/tensorboard/get_started#:~:text=TensorBoard%20is%20a%20tool%20for,dimensional%20space%2C%20and%20much%20more.)

[![License](https://img.shields.io/badge/License-MIT-blue)](#license)

</div>



## How to Use

### Prerequisite


Please make sure you have installed `python3.6` or higher versions of python.


### Install


```
git clone https://github.com/LightningCrashers/DyFEn
cd DyFEn
```
All dependencies will be handled using the command :

```pip install -r requirements.txt```


### Run

```
python3 -m scripts.ln_fee --algo PPO --tb_name PPO_tensorboard
python3 -m scripts.baselines --strategy static --node_index 71555
```

### Parameters
ln_fee:

| Parameter              | Default | choices                                      |
|------------------------|--------|----------------------------------------------|
| _--algo_               | PPO    | PPO, TRPO, SAC, TD3, A2C, DDPG     |
| _--node_index_           | 71555 | Arbitrary index in the data            |
| _--total_timesteps_    | 100000 | Arbitrary Integer                            |
| _--max_episode_length_ | 200    | Arbitrary Integer less than total_timesteps  |
| _--counts_             | [10, 10, 10] | List of Integers                             |
| _--amounts_            | [10000, 50000, 100000] | List of Integers |
| _--epsilons_           | [.6, .6, .6] | List of floats between 0 and 1               |


- You can modify the transaction sampling parameters by changing counts, amounts and epsilons
  - `counts` contains count of each transaction type. 
  - `amounts` contains amount of each transaction type in satoshi.
  - `epsilons` is the ratio of merchants in final sampling.
- Please note that length of counts, amounts and epsilons lists should be the same.
baselines:

| Parameter              | Default | choices                                      |
|------------------------|--------|----------------------------------------------|
| _--strategy_            | static | static, proportional, match_peer     |





You can check the results on tensorboard.

```
tensorboard --logdir plotting/tb_results/
```

## Trouble-shootings

If you are facing problems with tensorboard, run the command below in terminal :

```
python3 -m tensorboard.main --logdir plotting/tb_results/
```


## Citation



