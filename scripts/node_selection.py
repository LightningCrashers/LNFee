import simulator.preprocessing as preprocessing
import random


def select_random_node(directed_edges, conditions):
    channels_with_qualified_number_of_channels = get_channels_with_qualified_number_of_channels(directed_edges,
                                                                        conditions['n_channels_lower_bound'],
                                                                        conditions['n_channels_upper_bound'])

    channels_with_qualified_total_capacity = get_channels_with_qualified_total_capacity(directed_edges,
                                                                                        channels_with_qualified_number_of_channels,
                                                                                        conditions['total_capacity_lower_bound'],
                                                                                        conditions['total_capacity_upper_bound'])

    random_node_index = random.sample(channels_with_qualified_total_capacity, 15)
    return random_node_index


def get_channels_with_qualified_number_of_channels(directed_edges, n_channels_lower_bound, n_channels_upper_bound):
    candidate_channels = []
    for i in range(len(directed_edges)):
        src, trgs, channel_ids, number_of_channels = preprocessing.select_node(directed_edges, i)
        if n_channels_upper_bound > number_of_channels > n_channels_lower_bound:
            candidate_channels.append(i)
    return candidate_channels


def get_total_capacity(directed_edges, index):
    src = directed_edges.iloc[index]['src']
    trgs = directed_edges.loc[(directed_edges['src'] == src)]['trg']
    c = 0
    for trg in trgs:
        channel = directed_edges.loc[(directed_edges['src'] == src) & (directed_edges['trg'] == trg)]
        c += channel['capacity'].iloc[0]
    return c


def get_channels_with_qualified_total_capacity(directed_edges, channels_with_qualified_number_of_channels, total_capacity_lower_bound, total_capacity_upper_bound):
    candidate_channels = []
    for i in channels_with_qualified_number_of_channels:
        c = get_total_capacity(directed_edges, i)
        if total_capacity_lower_bound < c < total_capacity_upper_bound:
            candidate_channels.append(i)
    return candidate_channels


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Random node selection considering conditions on number of channels & total capacity')
    parser.add_argument('--data_path', default='../data/data.json')
    parser.add_argument('--n_channels_lower_bound', type=int, default=5)
    parser.add_argument('--n_channels_upper_bound', type=int, default=10)
    parser.add_argument('--total_capacity_lower_bound', type=int, default=15e6)
    parser.add_argument('--total_capacity_upper_bound', type=int, default=30e6)
    args = parser.parse_args()

    directed_edges_path = args.data_path
    directed_edges = preprocessing.get_directed_edges(directed_edges_path)

    conditions = {'n_channels_lower_bound': args.n_channels_lower_bound,
                  'n_channels_upper_bound': args.n_channels_upper_bound,
                  'total_capacity_lower_bound': args.total_capacity_lower_bound,
                  'total_capacity_upper_bound': args.total_capacity_upper_bound}

    random_node_index = select_random_node(directed_edges, conditions)
    print(random_node_index)


