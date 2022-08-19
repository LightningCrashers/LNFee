import networkx as nx

def get_excluded_total_fee(network_dictionary, path, excluded_src, excluded_trg) :
    alpha_bar = 0
    beta_bar = 0
    for i in range(len(path)-1):
      src = path[i]
      trg = path[i+1]
      if (src!=excluded_src) or (trg!=excluded_trg) :
        src_trg = network_dictionary[(src,trg)]
        alpha_bar += src_trg[1]
        beta_bar += src_trg[2]
    return alpha_bar, beta_bar

def get_min_balance(network_dictionary, path):
  min_balance = math.inf
  for i in range(len(path)-1):
      src = path[i]
      trg = path[i+1]
      src_trg = network_dictionary[(src,trg)]
      edge_balance = src_trg[0]
      if (edge_balance<min_balance)
        min_balance = edge_balance    
  return min_balance

def get_rebalancing_dict_values(G,node,source,target)
  paths = list(nx.all_simple_paths(G, source, target))
  values = []
  for path in paths:
    loop = path.insert(0,node)
    loop = path.append(node)
    max_flow_possible = get_min_balance(loop)
    alpha_bar, beta_bar = get_excluded_total_fee(path, source, target)
    val = (loop, max_flow_possible, alpha_bar, beta_bar)
    values.add(val)
  return values

def create_rebalancing_dict(node,neigbors_list) :
    pass

# TODO: channel_dictionary = {(src,channel_id):[trg, balance, alpha, beta, capacity]}

def create_sub_network(directed_edges,providers,src,trgs,channel_ids,capacities,initial_balances,radius):
  """creating network_dictionary, edges and providers for the local subgraph."""
  edges = initiate_balances(directed_edges)
  edges = set_channels_balances(edges,src,trgs,channel_ids,capacities,initial_balances)
  G = nx.from_pandas_edgelist(edges,source="src",target="trg",
                              edge_attr=['channel_id','capacity','fee_base_msat','fee_rate_milli_msat','balance'],create_using=nx.DiGraph())
  sub_nodes= get_neighbors(G,src,radius)
  print('sub_nodes : ',sub_nodes)
  sub_providers = list(set(sub_nodes) & set(providers))
  sub_graph = G.subgraph(sub_nodes)
  sub_edges = nx.to_pandas_edgelist(sub_graph)
  print('sub_edges : ')
  print(sub_edges)
  sub_edges = sub_edges.rename(columns={'source': 'src', 'target': 'trg'})
  network_dictionary = create_network_dictionary(sub_edges)
  #network_dictionary = {(src,trg):[balance,alpha,beta,capacity]}

  return network_dictionary, sub_nodes, sub_providers, sub_edges

def create_channel_dictionary(G):
    keys = list(zip(G["src"], G["channel_id"]))
    vals = [list(item) for item in zip([None]*len(G), G["trg"] ,G["fee_rate_milli_msat"],G['fee_base_msat'], G["capacity"])]
    channel_dictionary = dict(zip(keys,vals))
    for index,row in G.iterrows():
      src = row['src']
      trg = row['trg']
      channel_dictionary[(src,trg)][4] = row['balance']

    return channel_dictionary

def operate_multichannel_transaction(channel_dictionary, source, target, source_channel_id, target_channel_id, amount):
    src2 = channel_dictionary[(source, source_channel_id)][0]
    trg2 = channel_dictionary[(target, target_channel_id)][0]

    b_initial = channel_dictionary[(source, source_channel_id)][1]
    b_end = channel_dictionary[(trg2, target_channel_id)][1]
    if b_initial<=amount or b_end<=amount :
        return -1,None,0,0  #indicated channels dont have enough balances 

    graph = self.generate_graph(amount)  
    
    alpha_bar = 0
    beta_bar = 0
    reult_bit = -1

    if (not source in graph.nodes()) or (not target in graph.nodes()):
      return -2,None,0,0  # source and target are not in the graph
    path,result_bit = self.run_single_transaction(0,amount,src2,trg2,graph) 
    if result_bit == -1 :
      return -3,None,0,0  # no path between source and target
    elif result_bit == 1 :
      path.insert(0,source)
      path.append(target)
      alpha_bar,beta_bar = self.get_excluded_total_fee(path,source,src2)
      return 1, path, alpha_bar, beta_bar
