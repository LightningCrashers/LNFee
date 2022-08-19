import numpy as np
from env.multi_channel import FeeEnv
from simulator import preprocessing
from utils import load_data, make_env, get_fee_based_on_strategy, get_discounted_reward



def evaluate(strategy, env, env_params, gamma):
    directed_edges = preprocessing.get_directed_edges(env_params['data_path'])
    node_index = env_params['node_index']
    done = False
    state = env.reset()
    rewards = []
    while not done:
        action, rescale = get_fee_based_on_strategy(state, strategy, directed_edges, node_index)
        state, reward, done, info = env.step(action, rescale)
        state = state*1000
        rewards.append(reward)
        print(reward)

    discounted_reward = get_discounted_reward(rewards, gamma)
    return discounted_reward




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Baselines')
    parser.add_argument('--strategy', choices=['static', 'proportional', 'match_peer'], default='static', required=False)
    parser.add_argument('--data_path', default='../data/data.json')
    parser.add_argument('--merchants_path', default='../data/merchants.json')
    parser.add_argument('--fee_base_upper_bound', type=int, default=10000)
    parser.add_argument('--max_episode_length', type=int, default=200)
    parser.add_argument('--n_seed', type=int, default=1)  # 5
    parser.add_argument('--local_size', type=int, default=100)
    parser.add_argument('--node_index', type=int, default=71555)  # 97851
    parser.add_argument('--counts', default=[10, 10, 10], type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--amounts', default=[10000, 50000, 100000],
                        type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--epsilons', default=[.6, .6, .6], type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--manual_balance', default=True)
    parser.add_argument('--initial_balances', default= [214642, 589538, 9179347, 428493, 693709, 932820], type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--capacities', default= [221435, 1000000, 16777215, 1500000, 1000000, 1000000], type=lambda s: [int(item) for item in s.split(',')])

    args = parser.parse_args()

    # strategy = args.strategy
    env_params = {'data_path': args.data_path,
                  'merchants_path': args.merchants_path,
                  'node_index': args.node_index,
                  'fee_base_upper_bound': args.fee_base_upper_bound,
                  'max_episode_length': args.max_episode_length,
                  'local_size': args.local_size,
                  'counts': args.counts,
                  'amounts': args.amounts,
                  'epsilons': args.epsilons,
                  'manual_balance': args.manual_balance,
                  'initial_balances': args.initial_balances,
                  'capacities': args.capacities}


    strategy = args.strategy
    reward_list = []

    for s in range(args.n_seed):
        seed = np.random.randint(low=0, high=1000000)
        data = load_data(env_params['node_index'], env_params['data_path'], env_params['merchants_path'], env_params['local_size'],
                         env_params['manual_balance'], env_params['initial_balances'], env_params['capacities'])
        env = make_env(data, env_params, seed)
        discounted_reward = evaluate(strategy, env, env_params, gamma=0.99)
        reward_list.append(discounted_reward)


    import statistics
    mean = statistics.mean(reward_list)
    print('mean: ', mean)
