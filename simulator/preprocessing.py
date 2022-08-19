import networkx as nx
import pandas as pd
import json
import numpy as np
import random


def aggregate_edges(directed_edges):
    """aggregating multiedges"""
    grouped = directed_edges.groupby(["src", "trg"])
    directed_aggr_edges = grouped.agg({
        "capacity": "sum",
        "fee_base_msat": "mean",
        "fee_rate_milli_msat": "mean",
        "last_update": "max",
        "channel_id": "first",
        "disabled": "first",
        "min_htlc": "mean",
    }).reset_index()
    return directed_aggr_edges



def get_neighbors(G, src, local_size):
    """localising the network around the node"""

    neighbors = [src]

    for i in range(10):
        outer_list = []
        for neighbor in neighbors:
            inner_list = list(G.neighbors(neighbor))

            for v in inner_list:

               if len(neighbors) > local_size:
                  print('size of sub network: ', len(neighbors))
                  return set(neighbors)
               if v not in neighbor:
                  neighbors.append(v)


       

def initiate_balances(directed_edges, approach='half'):
    '''
    approach = 'random'
    approach = 'half'


    NOTE : This Function is written assuming that two side of channels are next to each other in directed_edges
    '''
    G = directed_edges[['src', 'trg', 'channel_id', 'capacity', 'fee_base_msat', 'fee_rate_milli_msat']]
    G = G.assign(balance=None)
    r = 0.5
    for index, row in G.iterrows():
        balance = 0
        cap = row['capacity']
        if index % 2 == 0:
            if approach == 'random':
                r = np.random.random()
            balance = r * cap
        else:
            balance = (1 - r) * cap
        G.at[index, "balance"] = balance

    return G


def set_channels_balances(edges, src, trgs, channel_ids, capacities, initial_balances):
    if (len(trgs) == len(capacities)) & (len(trgs) == len(initial_balances)):
        for i in range(len(trgs)):
            trg = trgs[i]
            capacity = capacities[i]
            initial_balance = initial_balances[i]
            index = edges.index[(edges['src'] == src) & (edges['trg'] == trg)]
            reverse_index = edges.index[(edges['src'] == trg) & (edges['trg'] == src)]

            edges.at[index[0], 'capacity'] = capacity
            edges.at[index[0], 'balance'] = initial_balance
            edges.at[reverse_index[0], 'capacity'] = capacity
            edges.at[reverse_index[0], 'balance'] = capacity - initial_balance

        return edges
    else:
        print("Error : Invalid Input Length")


def create_network_dictionary(G):
    keys = list(zip(G["src"], G["trg"]))
    vals = [list(item) for item in zip([None] * len(G), G["fee_rate_milli_msat"], G['fee_base_msat'], G["capacity"])]
    network_dictionary = dict(zip(keys, vals))
    for index, row in G.iterrows():
        src = row['src']
        trg = row['trg']
        network_dictionary[(src, trg)][0] = row['balance']

    return network_dictionary


def create_active_channels(network_dictionary, channels):
    # channels = [(src1,trg1),(src2,trg2),...]
    active_channels = dict()
    for (src, trg) in channels:
        active_channels[(src, trg)] = network_dictionary[(src, trg)]
        active_channels[(trg, src)] = network_dictionary[(trg, src)]
    return active_channels


def create_sub_network(directed_edges, providers, src, trgs, channel_ids, local_size, manual_balance=False, initial_balances = [], capacities=[]):
    """creating network_dictionary, edges and providers for the local subgraph."""
    edges = initiate_balances(directed_edges)
    if manual_balance:
        edges = set_channels_balances(edges, src, trgs, channel_ids, capacities, initial_balances)
    G = nx.from_pandas_edgelist(edges, source="src", target="trg",
                                edge_attr=['channel_id', 'capacity', 'fee_base_msat', 'fee_rate_milli_msat', 'balance'],
                                create_using=nx.DiGraph())
    sub_nodes = get_neighbors(G, src, local_size)
    sub_providers = list(set(sub_nodes) & set(providers))
    sub_graph = G.subgraph(sub_nodes)
    sub_edges = nx.to_pandas_edgelist(sub_graph)
    sub_edges = sub_edges.rename(columns={'source': 'src', 'target': 'trg'})
    network_dictionary = create_network_dictionary(sub_edges)
    # network_dictionary = {(src,trg):[balance,alpha,beta,capacity]}

    return network_dictionary, sub_nodes, sub_providers, sub_edges


def init_node_params(edges, providers, verbose=True):
    """Initialize source and target distribution of each node in order to draw transaction at random later."""
    G = nx.from_pandas_edgelist(edges, source="src", target="trg", edge_attr=["capacity"], create_using=nx.DiGraph())
    active_providers = list(set(providers).intersection(set(G.nodes())))
    active_ratio = len(active_providers) / len(providers)
    if verbose:
        print("Total number of possible providers: %i" % len(providers))
        print("Ratio of active providers: %.2f" % active_ratio)
    degrees = pd.DataFrame(list(G.degree()), columns=["pub_key", "degree"])
    total_capacity = pd.DataFrame(list(nx.degree(G, weight="capacity")), columns=["pub_key", "total_capacity"])
    node_variables = degrees.merge(total_capacity, on="pub_key")
    return node_variables, active_providers, active_ratio


def get_providers(providers_path):
    # The path should direct this to a json file containing providers
    with open(providers_path) as f:
        tmp_json = json.load(f)
    providers = []
    for i in range(len(tmp_json)):
        providers.append(tmp_json[i].get('pub_key'))
    return providers


def get_directed_edges(directed_edges_path):
    directed_edges = pd.read_json(directed_edges_path)
    directed_edges = aggregate_edges(directed_edges)
    return directed_edges


def select_node(directed_edges, src_index):
    src = directed_edges.iloc[src_index]['src']
    trgs = directed_edges.loc[(directed_edges['src'] == src)]['trg']
    channel_ids = directed_edges.loc[(directed_edges['src'] == src)]['channel_id']
    number_of_channels = len(trgs)
    return src, list(trgs), list(channel_ids), number_of_channels


def get_init_parameters(providers, directed_edges, src, trgs, channel_ids, channels, local_size, manual_balance, initial_balances, capacities):
    network_dictionary, nodes, sub_providers, sub_edges = create_sub_network(directed_edges, providers, src, trgs,
                                                                             channel_ids, local_size, manual_balance, initial_balances, capacities)
    active_channels = create_active_channels(network_dictionary, channels)
    try:
        node_variables, active_providers, active_ratio = init_node_params(sub_edges, sub_providers, verbose=True)
    except:
        print('zero providers!')

    balances = []
    capacities = []
    for trg in trgs:
        b = network_dictionary[(src, trg)][0]
        c = network_dictionary[(src, trg)][3]
        balances.append(b)
        capacities.append(c)

    return active_channels, network_dictionary, node_variables, active_providers, balances, capacities

def generate_transaction_types(number_of_transaction_types, counts, amounts, epsilons):
    transaction_types = []
    for i in range(number_of_transaction_types):
        transaction_types.append((counts[i], amounts[i], epsilons[i]))
    return transaction_types
