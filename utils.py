import numpy as np
import stable_baselines3
import sb3_contrib
from simulator import preprocessing
from env.multi_channel import FeeEnv


def make_agent(env, algo, device, tb_log_dir):
    policy = "MlpPolicy"
    # create model
    if algo == "PPO":
        from stable_baselines3 import PPO
        model = PPO(policy, env, verbose=1, device=device, tensorboard_log=tb_log_dir)
    elif algo == "TRPO":
        from sb3_contrib import TRPO
        model = TRPO(policy, env, verbose=1, device=device, tensorboard_log=tb_log_dir)
    elif algo == "SAC":
        from stable_baselines3 import SAC
        model = SAC(policy, env, verbose=1, device=device, tensorboard_log=tb_log_dir)
    elif algo == "DDPG":
        from stable_baselines3 import DDPG
        model = DDPG(policy, env, verbose=1, device=device, tensorboard_log=tb_log_dir)
    elif algo == "TD3":
        from stable_baselines3 import TD3
        model = TD3(policy, env, verbose=1, device=device, tensorboard_log=tb_log_dir)
    elif algo == "A2C":
        from stable_baselines3 import A2C
        model = A2C(policy, env, verbose=1, device=device, tensorboard_log=tb_log_dir)
    elif algo == "TQC":
        from sb3_contrib import TQC
        model = TQC(policy, env, verbose=1, device=device, tensorboard_log=tb_log_dir)
    elif algo == "ARS":
        from sb3_contrib import ARS
        model = ARS(policy, env, verbose=1, device=device, tensorboard_log=tb_log_dir)
    else:
        raise NotImplementedError()

    return model


def make_env(data, env_params, seed):
    assert len(env_params['counts']) == len(env_params['amounts']) and len(env_params['counts']) == len(
        env_params['epsilons']), "number of transaction types missmatch"
    env = FeeEnv(data, env_params['fee_base_upper_bound'], env_params['max_episode_length'],
                 len(env_params['counts']),
                 env_params['counts'], env_params['amounts'], env_params['epsilons'],
                 seed)

    return env


def load_data(node, directed_edges_path, providers_path, local_size, manual_balance, initial_balances, capacities):
    """
    :return:
    data = dict{src: node chosen for simulation  (default: int)
                trgs: nodes src is connected to  (default: list)
                channel_ids: channel ids of src node [NOT USED YET]  (default: list)
                initial_balances: initial distribution of capacity of each channel  (default: list)
                capacities: capacity of each channel  (default: list)
                node_variables: ???  (default: )
                providers: Merchants of the whole network  (default: ?)
                active_providers: Merchants of the local network around src  (default: ?)
                active_channels: channel which their balances are being updated each timestep  (default: ?)
                network_dictionary: whole network data  (default: dict)
            }
    """
    print('==================Loading Network Data==================')
    data = {}
    src_index = node
    subgraph_radius = 2
    data['providers'] = preprocessing.get_providers(providers_path)
    directed_edges = preprocessing.get_directed_edges(directed_edges_path)
    data['src'], data['trgs'], data['channel_ids'], n_channels = preprocessing.select_node(directed_edges, src_index)
    data['capacities'] = [153243, 8500000, 4101029, 5900000, 2500000, 7000000]
    data['initial_balances'] = [153243 / 2, 8500000 / 2, 4101029 / 2, 5900000 / 2, 2500000 / 2, 7000000 / 2]
    channels = []
    for trg in data['trgs']:
        channels.append((data['src'], trg))
    data['active_channels'], \
    data['network_dictionary'], \
    data['node_variables'], \
    data['active_providers'], \
    data['initial_balances'], \
    data['capacities'] = preprocessing.get_init_parameters(data['providers'],
                                                           directed_edges,
                                                           data['src'], data['trgs'],
                                                           data['channel_ids'],
                                                           channels,
                                                           local_size,
                                                           manual_balance, initial_balances, capacities)
    return data



def get_static_fee(directed_edges, node_index, number_of_channels):
    # action = get_original_fee(directed_edges, node_index)
    # action = get_mean_fee(directed_edges, number_of_channels)
    action = get_constant_fee(alpha=541.62316304877, beta=1760.82436708861, number_of_channels=number_of_channels)
    return action

def get_proportional_fee(state, number_of_channels, directed_edges, node_index):
    balances = state[0:number_of_channels]
    capacities = get_capacities(directed_edges, node_index)
    fee_rates = []
    for i in range(len(balances)):
        b = balances[i]
        c = capacities[i]
        f = -1 + 2*(1-(b/c))
        fee_rates.append(f)
    base_fees = [-1]*number_of_channels     # = zero after rescaling
    return fee_rates+base_fees

def get_match_peer_fee(directed_edges, node_index):
    action = get_peer_fee(directed_edges, node_index)
    return action


def get_fee_based_on_strategy(state, strategy, directed_edges, node_index):
    number_of_channels = get_number_of_channels(directed_edges, node_index)
    rescale = True
    if strategy == 'static':
        action = get_static_fee(directed_edges, node_index, number_of_channels)
        rescale = False
    elif strategy == 'proportional':
        action = get_proportional_fee(state, number_of_channels, directed_edges, node_index)
        action = np.array(action)
        rescale = True
    elif strategy == 'match_peer':
        action = get_match_peer_fee(directed_edges, node_index)
        rescale = False
    else:
        raise NotImplementedError
    return action, rescale


def get_mean_fee(directed_edges, number_of_channels):
    mean_alpha = directed_edges['fee_rate_milli_msat'].mean()
    mean_beta = directed_edges['fee_base_msat'].mean()
    return [mean_alpha]*number_of_channels + [mean_beta]*number_of_channels

def get_constant_fee(alpha, beta, number_of_channels):
    return [alpha]*number_of_channels + [beta]*number_of_channels

def get_original_fee(directed_edges, node_index):
    src = directed_edges.loc[node_index]['src']
    fee_rates = list(directed_edges[directed_edges['src'] == src]['fee_rate_milli_msat'])
    base_fees = list(directed_edges[directed_edges['src'] == src]['fee_base_msat'])
    return fee_rates + base_fees

def get_peer_fee(directed_edges, node_index):
    src = directed_edges.loc[node_index]['src']
    fee_rates = list(directed_edges[directed_edges['trg'] == src]['fee_rate_milli_msat'])
    base_fees = list(directed_edges[directed_edges['trg'] == src]['fee_base_msat'])
    return fee_rates + base_fees

def get_capacities(directed_edges, node_index):
    src = directed_edges.loc[node_index]['src']
    capacities = list(directed_edges[directed_edges['src'] == src]['capacity'])
    return capacities

def get_number_of_channels(directed_edges, node_index):
    src = directed_edges.loc[node_index]['src']
    number_of_channels = len(directed_edges[directed_edges['src'] == src])
    return number_of_channels


def get_discounted_reward(rewards, gamma):
    discounted_reward = 0
    for i in range(len(rewards)):
        coeff = pow(gamma, i)
        r = coeff*rewards[i]
        discounted_reward += r
    return discounted_reward


def load_model(algo, env_params, path):
    node_index = env_params['node_index']
    if algo == 'DDPG':
        from stable_baselines3 import DDPG
        model = DDPG.load(path=path)
    elif algo == 'PPO':
        from stable_baselines3 import PPO
        model = PPO.load(path=path)
    elif algo == 'TRPO':
        from sb3_contrib import TRPO
        model = TRPO.load(path=path)
    elif algo == 'TD3':
        from stable_baselines3 import TD3
        model = TD3.load(path=path)
    elif algo == 'A2C':
        from stable_baselines3 import A2C
        model = A2C.load(path=path)

    else:
        raise NotImplementedError

    return model



def load_localized_model(radius, path):
    from stable_baselines3 import PPO
    model = PPO.load(path=path)
    return model