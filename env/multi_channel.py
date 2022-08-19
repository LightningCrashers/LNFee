import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from simulator.simulator import simulator
from simulator.preprocessing import generate_transaction_types


class FeeEnv(gym.Env):
    """
    ### Description

    This environment corresponds to the LIGHTNING NETWORK simulation. A source node is chosen and a local network
    around that node with radius 2 is created and at each time step, a certain number of transitions are being simulated.

    ### Scales

    We are using the following scales for simulating the real world Lightning Network:

    - Fee Rate: msat                                      - Base Fee: msat
    - Transaction amounts: sat                            - Reward(income): msat
    - Capacity: sat                                       - Balance: sat

    ### Action Space

    The action is a `ndarray` with shape `(2*n_channel,)` which can take values `[0,upper bound]`
    indicating the fee rate and base fee of each channel starting from source node.

    | dim       | action                 | dim        | action                |
    |-----------|------------------------|------------|-----------------------|
    | 0         | fee rate channel 0     | 0+n_channel| fee base channel 0    |
    | ...       |        ...             | ...        |         ...           |
    | n_channel | fee rate last channel  | 2*n_channel| fee base last channel |

    ### Observation Space

    The observation is a `ndarray` with shape `(2*n_channel,)` with the values corresponding to the balance of each
    channel and also accumulative transaction amounts in each time steps.

    | dim       | observation            | dim        | observation                 |
    |-----------|------------------------|------------|-----------------------------|
    | 0         | balance channel 0      | 0+n_channel| sum trn amount channel 0    |
    | ...       |          ...           | ...        |            ...              |
    | n_channel | balance last channel   | 2*n_channel| sum trn amount last channel |

    ### Rewards

    Since the goal is to maximize the return in long term, reward is sum of incomes from fee payments of each channel.
    Reward scale is Sat in order to control the upperbound.

    ***Note:
    We are adding the income from each payment to balance of the corresponding channel.
    """

    def __init__(self, data, fee_base_upper_bound, max_episode_length, number_of_transaction_types, counts, amounts, epsilons, seed):
        # Source node
        self.src = data['src']
        self.trgs = data['trgs']
        self.n_channel = len(self.trgs)
        print('actione dim:', 2 * self.n_channel)

        # Base fee and fee rate for each channel of src
        self.action_space = spaces.Box(low=-1, high=+1, shape=(2 * self.n_channel,), dtype=np.float32)
        self.fee_rate_upper_bound = 1000
        self.fee_base_upper_bound = fee_base_upper_bound

        # Balance and transaction amount of each channel
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2 * self.n_channel,), dtype=np.float32)

        # Initial values of each channel
        self.initial_balances = data['initial_balances']
        self.capacities = data['capacities']
        self.state = np.append(self.initial_balances, np.zeros(shape=(self.n_channel,)))

        self.time_step = 0
        self.max_episode_length = max_episode_length
        self.balance_ratio = 0.1

        # Simulator
        transaction_types = generate_transaction_types(number_of_transaction_types, counts, amounts,
                                                       epsilons)
        self.simulator = simulator(src=data['src'],
                                   trgs=data['trgs'],
                                   channel_ids=data['channel_ids'],
                                   active_channels=data['active_channels'],
                                   network_dictionary=data['network_dictionary'],
                                   merchants=data['providers'],
                                   transaction_types=transaction_types,
                                   node_variables=data['node_variables'],
                                   active_providers=data['active_providers'],
                                   fixed_transactions=False)

        self.seed(seed)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action, rescale=True):
        # Rescaling the action vector
        if rescale:
            action[0:self.n_channel] = .5 * self.fee_rate_upper_bound * action[0:self.n_channel] + \
                                       .5 * self.fee_rate_upper_bound
            action[self.n_channel:2 * self.n_channel] = .5 * self.fee_base_upper_bound * action[
                                                                                         self.n_channel:2 * self.n_channel] + \
                                                        .5 * self.fee_base_upper_bound

        # Running simulator for a certain time interval
        balances, transaction_amounts, transaction_numbers = self.simulate_transactions(action)
        self.time_step += 1

        reward = 1e-6 * np.sum(np.multiply(action[0:self.n_channel], transaction_amounts) + \
                        np.multiply(action[self.n_channel:2 * self.n_channel], transaction_numbers))

        info = {'TimeLimit.truncated': True if self.time_step >= self.max_episode_length else False}

        done = self.time_step >= self.max_episode_length

        self.state = np.append(balances, transaction_amounts)/1000

        return self.state, reward, done, info

    def simulate_transactions(self, action):
        self.simulator.set_channels_fees(action)

        output_transactions_dict = self.simulator.run_simulation(action)
        balances, transaction_amounts, transaction_numbers = self.simulator.get_simulation_results(action,
                                                                                                   output_transactions_dict)

        return balances, transaction_amounts, transaction_numbers

    def reset(self):
        print('episode ended!')
        self.time_step = 0
        self.state = np.append(self.initial_balances, np.zeros(shape=(self.n_channel,)))

        return np.array(self.state, dtype=np.float64)

