import numpy as np
from env.multi_channel import FeeEnv
from simulator import preprocessing
from utils import load_data, make_env, get_discounted_reward, load_localized_model


def evaluate(model, env, gamma):
    done = False
    state = env.reset()
    rewards = []
    while not done:
        action, _state = model.predict(state)
        action = np.array(action)
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        print(reward)

    discounted_reward = get_discounted_reward(rewards, gamma)
    return discounted_reward




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Localization')
    parser.add_argument('--log_dir', default='')
    parser.add_argument('--data_path', default='../data/data.json')
    parser.add_argument('--merchants_path', default='../data/merchants.json')
    parser.add_argument('--fee_base_upper_bound', type=int, default=10000)
    parser.add_argument('--max_episode_length', type=int, default=200)
    parser.add_argument('--n_seed', type=int, default=5)  # 5
    parser.add_argument('--local_size', type=int, default=100)
    parser.add_argument('--local_size_option', choices=[100, 250, 500], default=500)
    parser.add_argument('--node_index', type=int, default=97851)  # 97851
    parser.add_argument('--counts', default=[10, 10, 10], type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--amounts', default=[10000, 50000, 100000],
                        type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--epsilons', default=[.6, .6, .6], type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--manual_balance', default=False)
    parser.add_argument('--initial_balances', default=[],
                        type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--capacities', default=[], type=lambda s: [int(item) for item in s.split(',')])

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



    #local_size_option = args.local_size_option
    radius_options = [100, 250, 500]
    reward_dict = dict()
    for radius in radius_options:
        reward_dict[radius] = []

    for s in range(args.n_seed):
        seed = np.random.randint(low=0, high=1000000)
        data = load_data(env_params['node_index'], env_params['data_path'], env_params['merchants_path'], env_params['local_size'],
                         env_params['manual_balance'], [], [])
        for radius in radius_options:
            env = make_env(data, env_params, seed)
            model = load_localized_model(radius)
            model.set_env(env)

            discounted_reward = evaluate(model, env, gamma=0.99)
            reward_dict[radius].append(discounted_reward)


    import statistics
    mean_reward_dict = dict()
    for radius in radius_options:
        mean_reward_dict[radius] = statistics.mean(reward_dict[radius])

    print('_____________________________________________________')
    print(mean_reward_dict)
