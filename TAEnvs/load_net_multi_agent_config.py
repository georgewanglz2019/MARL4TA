import os
import numpy as np
import pandas as pd
import networkx as nx
import copy as cp
import shutil
import pickle
import sys
import os
import matplotlib.pyplot as plt


class Config():
    def __init__(self, net_df, demand_df, all_od_paths, action_space, batch=None):
        self.net_df = net_df
        self.demand_df = demand_df
        self.all_od_paths = all_od_paths
        self.action_space = action_space
        self.batch = batch


def load_SiouxFalls_so_action_set_multi_agent_config(file_path, remove_single_route_od_pairs=True):

    ################################### Loading Net data ###################################
    # ------------- read net file -------------
    sioux_falls_net = pd.read_csv(os.path.join(file_path, 'SiouxFalls_net.tntp'), skiprows=8, sep='\t').drop(['~', ';'],
                                                                                                             axis=1)
    sioux_falls_net['edge'] = sioux_falls_net.index + 1

    node_coord = pd.read_csv(os.path.join(file_path, 'SiouxFalls_node.tntp'), sep='\t').drop([';'],
                                                                                             axis=1)  # Actual Sioux Falls coordinate
    node_xy = pd.read_csv(os.path.join(file_path, 'SiouxFalls_node_xy.tntp'),
                          sep='\t')  # X,Y position for good visualization
    node_xy['X_normalized'] = (node_xy['X'] - min(node_xy['X']))/(max(node_xy['X']) - min(node_xy['X']))
    node_xy['Y_normalized'] = (node_xy['Y'] - min(node_xy['Y'])) / (max(node_xy['Y']) - min(node_xy['Y']))

    node_xy_dict = {}   # {1: array([0., 1.]),  2: array([0.72972973, 1.]),...}
    for i in range(len(node_xy)):
        node_i = node_xy.Node[i]
        xy_array = node_xy.loc[node_xy.Node == node_i, ['X_normalized', 'Y_normalized']].values.flatten()
        node_xy_dict[node_i] = xy_array
    #node_xy_dict

    sioux_falls_df = sioux_falls_net.copy()
    sioux_falls_df['flow'] = 0
    sioux_falls_df['cost'] = sioux_falls_df['free_flow_time'] * (
            1 + 0.15 * ((sioux_falls_df['flow'] / sioux_falls_df['capacity']) ** 4))
    sioux_falls_df['weight'] = sioux_falls_df['free_flow_time']
    sioux_falls_df['_'] = '_'
    sioux_falls_df['od'] = sioux_falls_df["init_node"].astype(str) + sioux_falls_df['_'] + sioux_falls_df[
        "term_node"].astype(str)
    sioux_falls_df = sioux_falls_df.drop(['_'], axis=1)

    # coordinate position
    pos_coord = dict([(i, (a, b)) for i, a, b in zip(node_coord.Node, node_coord.X, node_coord.Y)])

    # read demand file
    sioux_falls_demand_df = pd.read_csv(os.path.join(file_path, 'SiouxFalls_od.csv'))
    sioux_falls_demand_df.columns = ['init_node', 'term_node', 'demand']

    od_demand_G = nx.from_pandas_edgelist(sioux_falls_demand_df, 'init_node', 'term_node', ['demand'],
                                          create_using=nx.DiGraph())

    demand_adj = nx.to_numpy_array(od_demand_G,
                                   nodelist=[i + 1 for i in range(max(sioux_falls_demand_df.init_node.unique()))],
                                   weight='demand')

    import pickle


    learned_chosen_routes_count_pkl = open(os.path.join(file_path, 'route_set_dict_500.pkl'), 'rb')
    so_used_routes = pickle.load(learned_chosen_routes_count_pkl)
    print('len(so_used_routes):', len(so_used_routes))

    rl_agent_od = []  # rl-agents <--> od-pairs
    one_route_od = []

    max_so_routes_num = 0
    for od, r_list in so_used_routes.items():
        route_choices = len(r_list)
        if route_choices == 1:
            one_route_od.append(od)
            if remove_single_route_od_pairs:
                pass
            else:
                rl_agent_od.append(od)
        else:
            rl_agent_od.append(od)

        if route_choices > max_so_routes_num:
            max_so_routes_num = route_choices
    #print('max_so_routes_num=', max_so_routes_num)  # max_so_routes_num = 11 in SiouxFall net
    print('od number:', len(so_used_routes))
    print('rl_agent_od number:', len(rl_agent_od))
    print('one_route_od number:', len(one_route_od))

    # ------------- get shortest paths -------------
    def k_shortest_paths(G, source, target, k=1, weight='weight'):
        # G is a networkx graph.
        # source and target are the labels for the source and target of the path.
        # k is the amount of desired paths.
        # weight = 'weight' assumes a weighed graph. If this is undesired, use weight = None.

        A = [nx.dijkstra_path(G, source, target, weight='weight')]
        A_len = [sum([G[A[0][l]][A[0][l + 1]]['weight'] for l in range(len(A[0]) - 1)])]
        B = []

        for i in range(1, k):
            for j in range(0, len(A[-1]) - 1):
                Gcopy = cp.deepcopy(G)
                spurnode = A[-1][j]
                rootpath = A[-1][:j + 1]
                for path in A:
                    if rootpath == path[0:j + 1]:  # and len(path) > j?
                        if Gcopy.has_edge(path[j], path[j + 1]):
                            Gcopy.remove_edge(path[j], path[j + 1])
                        if Gcopy.has_edge(path[j + 1], path[j]):
                            Gcopy.remove_edge(path[j + 1], path[j])
                for n in rootpath:
                    if n != spurnode:
                        Gcopy.remove_node(n)
                try:
                    spurpath = nx.dijkstra_path(Gcopy, spurnode, target, weight='weight')
                    totalpath = rootpath + spurpath[1:]
                    if totalpath not in B:
                        B += [totalpath]
                except nx.NetworkXNoPath:
                    continue
            if len(B) == 0:
                break
            lenB = [sum([G[path[l]][path[l + 1]]['weight'] for l in range(len(path) - 1)]) for path in B]
            B = [p for _, p in sorted(zip(lenB, B))]
            A.append(B[0])
            A_len.append(sorted(lenB)[0])
            B.remove(B[0])

        return A, A_len

    ksp_paths_dict = {}
    nx_graph_G = nx.from_pandas_edgelist(sioux_falls_df, 'init_node', 'term_node', ['weight'],
                                         create_using=nx.DiGraph())
    k = 20


    sf_od_pairs_l = list(so_used_routes.keys())
    for init_n in sioux_falls_df.init_node.unique():
        for term_n in sioux_falls_df.term_node.unique():
            #if init_n != term_n:
            init_term_ = str(init_n) + '_' + str(term_n)
            if (init_n != term_n) and (init_term_ in sf_od_pairs_l):
                paths, lengths = k_shortest_paths(
                    nx_graph_G, init_n, term_n, k=k, weight='weight')
                ksp_paths_dict[init_term_] = paths


    routes_num_threshold = 6
    print('routes_num_threshold=', routes_num_threshold)  # number of routes should <= routes_num_threshold
    for o_d, paths in so_used_routes.items():
        num_learned_paths = len(paths)
        if num_learned_paths < routes_num_threshold:
            num_gap = routes_num_threshold - num_learned_paths

            for nn in range(num_gap):
                # so_used_routes[o_d].append(ksp_paths_dict[o_d][-1])
                so_used_routes[o_d].append(paths[0])

        elif num_learned_paths > routes_num_threshold:
            so_used_routes[o_d] = so_used_routes[o_d][:routes_num_threshold]


    def edges_from_path(s_path):
        # s_path [1, 3, 4, 5, 6, 2]
        edges = []
        for i in range(len(s_path) - 1):
            o = str(s_path[i])
            d = str(s_path[i + 1])
            edges.append(o + '_' + d)
        return edges  # ['1_3', '3_4', '4_5', '5_6', '6_2']


    route_info = {}
    current_id = 0


    #print(len())
    for od, routes in so_used_routes.items():
        route_info[od] = {}
        route_info[od]['id_list'] = []
        route_info[od]['route_list'] = []
        route_info[od]['edge_list'] = []
        route_info[od]['link_embedding'] = [] 
        route_info[od]['free_flow_tt'] = []
        route_info[od]['n_links'] = []
        for route in routes:

            # route_ids[od]['route_id'] = current_id
            route_info[od]['id_list'].append(current_id)
            route_info[od]['route_list'].append(route)
            route_edges = edges_from_path(route)
            route_info[od]['edge_list'].append(route_edges)
            link_embeddings = sioux_falls_df.od.isin(route_edges).values.astype(int)
            route_info[od]['link_embedding'].append(link_embeddings)
            route_info[od]['free_flow_tt'].append(np.dot(link_embeddings, sioux_falls_df.free_flow_time))
            route_info[od]['n_links'].append(len(route))
            current_id += 1
    n_routes = current_id
    #route_info['n_routes'] = current_id

    # ------------- get od demand embeddings -------------
    sioux_falls_demand_df['_'] = '_'
    sioux_falls_demand_df['od'] = sioux_falls_demand_df.init_node.astype(str) + sioux_falls_demand_df[
        '_'] + sioux_falls_demand_df.term_node.astype(str)

    n_nodes = max(max(sioux_falls_demand_df.init_node.values), max(sioux_falls_demand_df.term_node.values))
    od_demand_embeddings = {}
    for i in range(len(sioux_falls_demand_df)):
        embeddings = np.zeros(n_nodes)
        demand = sioux_falls_demand_df.loc[i, 'demand']
        embeddings[sioux_falls_demand_df.loc[i, 'init_node'] - 1] = -demand
        embeddings[sioux_falls_demand_df.loc[i, 'term_node'] - 1] = demand
        od_demand_embeddings[sioux_falls_demand_df.od[i]] = embeddings

    # ------------- get agent id map to od-pair -------------
    agent_ids = {}  #  {0: '21_1',..., 527: '14_24'}
    for i in range(len(rl_agent_od)):
        agent_ids[i] = rl_agent_od[i]

    print('len(agent_ids):', len(agent_ids))


    # ------------- get graph features: degree, centrality -------------
    sioux_falls_df['edge_idx'] = sioux_falls_df.index
    edge_idx_map = dict(zip(sioux_falls_df.od, sioux_falls_df.edge_idx))
    idx_edge_map = dict(zip(sioux_falls_df.edge_idx, sioux_falls_df.od))
    net_edges = sioux_falls_df.od.values

    edge_as_nodes = []
    for o_edge in net_edges: 
        o_edge_o, o_edge_d = o_edge.split('_')
        for d_edge in net_edges:
            d_edge_o, d_edge_d = d_edge.split('_')
            if (o_edge_d == d_edge_o) and (o_edge_o != d_edge_d): 
                new_gat_edge = [o_edge, d_edge]
                if new_gat_edge not in edge_as_nodes:
                    edge_as_nodes.append(new_gat_edge)
    #print(edge_as_nodes)

    df_gat = pd.DataFrame({'init_node': np.array(edge_as_nodes)[:, 0], 'terminate_node': np.array(edge_as_nodes)[:, 1]})
    df_gat['init_idx'] = df_gat['init_node'].map(edge_idx_map)
    df_gat['terminate_idx'] = df_gat['terminate_node'].map(edge_idx_map)

    G_gat_two_node = nx.from_pandas_edgelist(df_gat, 'init_idx', 'terminate_idx',
                                             ['init_node', 'terminate_node'], create_using=nx.DiGraph())
    #### get degree
    degree_dict = {}
    for k, v in G_gat_two_node.degree():
        degree_dict[idx_edge_map[k]] = v

    #### get centrality
    betweenness_centrality = nx.betweenness_centrality(G_gat_two_node)
    #print(betweenness_centrality)
    centrality_dict = {}
    for k, v in betweenness_centrality.items():
        centrality_dict[idx_edge_map[k]] = v

    #### add degree_norm and centrality_norm to sioux_falls_df
    sioux_falls_df['degree'] = sioux_falls_df.od.map(degree_dict)
    sioux_falls_df['centrality'] = sioux_falls_df.od.map(centrality_dict)
    sioux_falls_df['degree_norm'] = (sioux_falls_df['degree'] - sioux_falls_df['degree'].min()) / (
                sioux_falls_df['degree'].max() - sioux_falls_df['degree'].min())
    sioux_falls_df['centrality_norm'] = (sioux_falls_df['centrality'] - sioux_falls_df['centrality'].min()) / (
                sioux_falls_df['centrality'].max() - sioux_falls_df['centrality'].min())

    sioux_falls_df.head()

    # ------------- Get config -------------
    sf_so_config = Config(net_df=sioux_falls_df, demand_df=sioux_falls_demand_df, all_od_paths=so_used_routes,
                          action_space=routes_num_threshold)
    #sf_so_config.so_routes_dist_dict = so_routes_dist_dict
    sf_so_config.node_xy = node_xy
    sf_so_config.node_xy_dict = node_xy_dict  # {1: array([0., 1.]),  2: array([0.72972973, 1.]),...}
    sf_so_config.route_info = route_info
    sf_so_config.n_routes = n_routes
    sf_so_config.od_demand_embeddings = od_demand_embeddings
    sf_so_config.remove_single_route_od_pairs = remove_single_route_od_pairs
    sf_so_config.rl_agent_od = rl_agent_od
    sf_so_config.one_route_od = one_route_od
    sf_so_config.agent_ids = agent_ids
    #print(agent_ids)
    sf_so_config.max_env_iter_num = 50
    sf_so_config.net_name = 'siouxfalls'


    print('SiouxFalls_demand_df:', sioux_falls_demand_df.head(5))


    return sf_so_config


def load_Anaheim_so_action_set_multi_agent_config(file_path, remove_single_route_od_pairs=True):

    ################################### Loading data ###################################
    # ------------- 1. read net file -------------
    Anaheim_netfile = os.path.join(file_path, 'Anaheim_net.tntp')
    Anaheim_net_df = pd.read_csv(Anaheim_netfile, skiprows=8, sep='\t')

    trimmed = [s.strip().lower() for s in Anaheim_net_df.columns]
    Anaheim_net_df.columns = trimmed

    # And drop the silly first andlast columns
    Anaheim_net_df.drop(['~', ';'], axis=1, inplace=True)
    Anaheim_net_df['edge'] = Anaheim_net_df.index + 1

    # sioux_falls_df = sioux_falls_net.copy()
    Anaheim_net_df['flow'] = 0
    Anaheim_net_df['cost'] = Anaheim_net_df['free_flow_time']  # start with fftt
    Anaheim_net_df['weight'] = Anaheim_net_df['free_flow_time']
    Anaheim_net_df['_'] = '_'
    Anaheim_net_df['od'] = Anaheim_net_df["init_node"].astype(str) + Anaheim_net_df['_'] + Anaheim_net_df[
        "term_node"].astype(str)
    Anaheim_net_df = Anaheim_net_df.drop(['_'], axis=1)

    # ------------- 2. read coordinate file -------------
    Anaheim_node_xy = pd.read_csv(os.path.join(file_path, 'anaheim_nodes_coordinate.csv'))
    Anaheim_node_xy['X_normalized'] = (Anaheim_node_xy['X'] - min(Anaheim_node_xy['X'])) / (
                max(Anaheim_node_xy['X']) - min(Anaheim_node_xy['X']))
    Anaheim_node_xy['Y_normalized'] = (Anaheim_node_xy['Y'] - min(Anaheim_node_xy['Y'])) / (
                max(Anaheim_node_xy['Y']) - min(Anaheim_node_xy['Y']))

    Anaheim_node_xy_dict = {}  # {1: array([0., 1.]),  2: array([0.72972973, 1.]),...}
    for i in range(len(Anaheim_node_xy)):
        node_i = Anaheim_node_xy.node[i]
        xy_array = Anaheim_node_xy.loc[
            Anaheim_node_xy.node == node_i, ['X_normalized', 'Y_normalized']].values.flatten()
        Anaheim_node_xy_dict[node_i] = xy_array

    # ------------- 3. read demand df file -------------

    # Function to parse the tntp file content into a pandas DataFrame
    def parse_tntp_file(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        data = []
        origin = None

        for line in lines:
            line = line.strip()
            if line.startswith("Origin"):
                # Extract the origin node
                origin = int(line.split()[1])
            elif ":" in line:
                # Extract term_node and demand
                term_data = line.split(";")
                for term in term_data:
                    if term.strip():
                        term_node, demand = term.split(":")
                        term_node = int(term_node.strip())
                        demand = float(demand.strip())
                        od = f"{origin}_{term_node}"
                        data.append([origin, term_node, demand, '_', od])
        # Create a DataFrame
        df = pd.DataFrame(data, columns=['init_node', 'term_node', 'demand', '_', 'od'])
        return df

    Anaheim_demand_df = parse_tntp_file(os.path.join(file_path, "Anaheim_trips.tntp"))

    # ------------- 4. get od demand embedding -------------
    # od demand embedding
    n_nodes = max(max(Anaheim_demand_df.init_node.values), max(Anaheim_demand_df.term_node.values))
    print('n_nodes=', n_nodes)
    od_demand_embeddings = {}
    for i in range(len(Anaheim_demand_df)):
        embeddings = np.zeros(n_nodes)
        demand = Anaheim_demand_df.loc[i, 'demand']
        embeddings[Anaheim_demand_df.loc[i, 'init_node'] - 1] = -demand
        embeddings[Anaheim_demand_df.loc[i, 'term_node'] - 1] = demand
        od_demand_embeddings[Anaheim_demand_df.od[i]] = embeddings

    # ------------- 5. get route info -------------
    import pickle
    learned_chosen_routes_count_pkl = open(os.path.join(file_path, 'Anaheim_SO_route_set_dict_500.pkl'), 'rb')
    so_used_routes = pickle.load(learned_chosen_routes_count_pkl)
    print('len(so_used_routes):', len(so_used_routes))

    rl_agent_od = []  # rl-agents <--> od-pairs
    one_route_od = []

    max_so_routes_num = 0
    for od, r_list in so_used_routes.items():
        route_choices = len(r_list)
        if route_choices == 1:
            one_route_od.append(od)
            if remove_single_route_od_pairs:
                pass
            else:
                rl_agent_od.append(od)
        else:
            rl_agent_od.append(od)

        if route_choices > max_so_routes_num:
            max_so_routes_num = route_choices
    #print('max_so_routes_num=', max_so_routes_num)  # max_so_routes_num = 11 in SiouxFall net
    print('od number:', len(so_used_routes))
    print('rl_agent_od number:', len(rl_agent_od))
    print('one_route_od number:', len(one_route_od))


    routes_num_threshold = 6
    print('routes_num_threshold=', routes_num_threshold)  # number of routes should <= routes_num_threshold
    for o_d, paths in so_used_routes.items():
        num_learned_paths = len(paths)
        if num_learned_paths < routes_num_threshold:
            num_gap = routes_num_threshold - num_learned_paths

            for nn in range(num_gap):
                # so_used_routes[o_d].append(ksp_paths_dict[o_d][-1])
                so_used_routes[o_d].append(paths[0])

        elif num_learned_paths > routes_num_threshold:
            so_used_routes[o_d] = so_used_routes[o_d][:routes_num_threshold]

    def edges_from_path(s_path):
        # s_path [1, 3, 4, 5, 6, 2]
        edges = []
        for i in range(len(s_path) - 1):
            o = str(s_path[i])
            d = str(s_path[i + 1])
            edges.append(o + '_' + d)
        return edges  # ['1_3', '3_4', '4_5', '5_6', '6_2']

    route_info = {}
    current_id = 0

    #print(len())
    for od, routes in so_used_routes.items():
        route_info[od] = {}
        route_info[od]['id_list'] = []
        route_info[od]['route_list'] = []
        route_info[od]['edge_list'] = []
        route_info[od]['link_embedding'] = [] 
        route_info[od]['free_flow_tt'] = []
        route_info[od]['n_links'] = []
        for route in routes:
            # route_ids[od]['route_id'] = current_id
            route_info[od]['id_list'].append(current_id)
            route_info[od]['route_list'].append(route)
            route_edges = edges_from_path(route)
            route_info[od]['edge_list'].append(route_edges)
            link_embeddings = Anaheim_net_df.od.isin(route_edges).values.astype(int)
            route_info[od]['link_embedding'].append(link_embeddings)
            route_info[od]['free_flow_tt'].append(np.dot(link_embeddings, Anaheim_net_df.free_flow_time))
            route_info[od]['n_links'].append(len(route))
            current_id += 1
    n_routes = current_id
    #route_info['n_routes'] = current_id

    # ------------- get agent id map to od-pair -------------
    agent_ids = {}  #  {0: '21_1',..., 527: '14_24'}
    for i in range(len(rl_agent_od)):
        agent_ids[i] = rl_agent_od[i]

    print('len(agent_ids):', len(agent_ids))

    # ------------- 6. get graph features -------------
    # ------------- get graph features: degree, centrality -------------
    Anaheim_net_df['edge_idx'] = Anaheim_net_df.index
    edge_idx_map = dict(zip(Anaheim_net_df.od, Anaheim_net_df.edge_idx))
    idx_edge_map = dict(zip(Anaheim_net_df.edge_idx, Anaheim_net_df.od))
    net_edges = Anaheim_net_df.od.values


    edge_as_nodes = []
    for o_edge in net_edges:
        o_edge_o, o_edge_d = o_edge.split('_')
        for d_edge in net_edges:
            d_edge_o, d_edge_d = d_edge.split('_')
            if (o_edge_d == d_edge_o) and (o_edge_o != d_edge_d):
                new_gat_edge = [o_edge, d_edge]
                if new_gat_edge not in edge_as_nodes:
                    edge_as_nodes.append(new_gat_edge)
    # print(edge_as_nodes)

    df_gat = pd.DataFrame({'init_node': np.array(edge_as_nodes)[:, 0], 'terminate_node': np.array(edge_as_nodes)[:, 1]})
    # print('df_gat.head()=', df_gat.head())
    df_gat['init_idx'] = df_gat['init_node'].map(edge_idx_map)
    df_gat['terminate_idx'] = df_gat['terminate_node'].map(edge_idx_map)
    print('df_gat.head()=', df_gat.head())

    G_gat_two_node = nx.from_pandas_edgelist(df_gat, 'init_idx', 'terminate_idx',
                                             ['init_node', 'terminate_node'], create_using=nx.DiGraph())
    #### get degree
    degree_dict = {}
    for k, v in G_gat_two_node.degree():
        degree_dict[idx_edge_map[k]] = v
    # print('degree_dict=', degree_dict)

    #### get centrality
    betweenness_centrality = nx.betweenness_centrality(G_gat_two_node)
    # print(betweenness_centrality)
    centrality_dict = {}
    for k, v in betweenness_centrality.items():
        centrality_dict[idx_edge_map[k]] = v

    #### add degree_norm and centrality_norm to sioux_falls_df
    Anaheim_net_df['degree'] = Anaheim_net_df.od.map(degree_dict)
    Anaheim_net_df['centrality'] = Anaheim_net_df.od.map(centrality_dict)
    Anaheim_net_df['degree_norm'] = (Anaheim_net_df['degree'] - Anaheim_net_df['degree'].min()) / (
            Anaheim_net_df['degree'].max() - Anaheim_net_df['degree'].min())
    Anaheim_net_df['centrality_norm'] = (Anaheim_net_df['centrality'] - Anaheim_net_df['centrality'].min()) / (
            Anaheim_net_df['centrality'].max() - Anaheim_net_df['centrality'].min())


    # ------------- Get config -------------
    Anaheim_so_config = Config(net_df=Anaheim_net_df, demand_df=Anaheim_demand_df, all_od_paths=so_used_routes,
                          action_space=routes_num_threshold)
    #sf_so_config.so_routes_dist_dict = so_routes_dist_dict
    Anaheim_so_config.node_xy = Anaheim_node_xy
    Anaheim_so_config.node_xy_dict = Anaheim_node_xy_dict  # {1: array([0., 1.]),  2: array([0.72972973, 1.]),...}
    Anaheim_so_config.route_info = route_info
    Anaheim_so_config.n_routes = n_routes
    Anaheim_so_config.od_demand_embeddings = od_demand_embeddings
    Anaheim_so_config.remove_single_route_od_pairs = remove_single_route_od_pairs
    Anaheim_so_config.rl_agent_od = rl_agent_od
    Anaheim_so_config.one_route_od = one_route_od
    Anaheim_so_config.agent_ids = agent_ids
    #print(agent_ids)
    Anaheim_so_config.max_env_iter_num = 50
    Anaheim_so_config.net_name = 'Anaheim'


    print('SiouxFalls_demand_df:', Anaheim_demand_df.head(5))


    return Anaheim_so_config


def load_OW_ksp_multi_agent_config(file_path, ksp=6, remove_single_route_od_pairs=False, vehs_per_agent=1):
    ################################### Loading data ###################################
    # ------------- 1. read net file -------------
    OW_df = pd.read_csv(os.path.join(file_path, 'OW_net.csv'), sep=',')
    OW_df['edge'] = OW_df.index + 1
    OW_df['flow'] = 0
    OW_df['cost'] = OW_df['free_flow_time']
    OW_df['weight'] = OW_df['free_flow_time']
    OW_df['_'] = '_'
    OW_df['od'] = OW_df["init_node"].astype(str) + OW_df['_'] + OW_df["term_node"].astype(str)
    OW_df = OW_df.drop(['_'], axis=1)

    # ------------- 2. read coordinate file -------------
    node_xy = pd.read_csv(os.path.join(file_path,'OW_Node_XY.csv'))
    node_xy['X_normalized'] = (node_xy['X'] - min(node_xy['X']))/(max(node_xy['X']) - min(node_xy['X']))
    node_xy['Y_normalized'] = (node_xy['Y'] - min(node_xy['Y'])) / (max(node_xy['Y']) - min(node_xy['Y']))

    node_xy_dict = {}   # {1: array([0., 1.]),  2: array([0.72972973, 1.]),...}
    for i in range(len(node_xy)):
        node_i = node_xy.Node[i]
        xy_array = node_xy.loc[node_xy.Node == node_i, ['X_normalized', 'Y_normalized']].values.flatten()
        node_xy_dict[node_i] = xy_array
    print('OW_Node_XY_dict:', node_xy_dict)

    # ------------- 3. read demand df file -------------
    OW_demand_df = pd.read_csv(os.path.join(file_path, 'OW_trips.csv'), sep=',')
    OW_demand_df['_'] = '_'
    OW_demand_df['od'] = OW_demand_df.init_node.astype(str) + OW_demand_df['_'] + OW_demand_df.term_node.astype(str)

    # ------------- 4. get od demand embedding -------------
    # od demand embedding
    n_nodes = max(max(OW_demand_df.init_node.values), max(OW_demand_df.term_node.values))
    print('n_nodes=', n_nodes)
    od_demand_embeddings = {}
    for i in range(len(OW_demand_df)):
        embeddings = np.zeros(n_nodes)
        demand = OW_demand_df.loc[i, 'demand']
        embeddings[OW_demand_df.loc[i, 'init_node'] - 1] = -demand
        embeddings[OW_demand_df.loc[i, 'term_node'] - 1] = demand
        od_demand_embeddings[OW_demand_df.od[i]] = embeddings

    # ------------- 5. get route info -------------
    def k_shortest_paths(G, source, target, k=1, weight='weight'):
        # G is a networkx graph.
        # source and target are the labels for the source and target of the path.
        # k is the amount of desired paths.
        # weight = 'weight' assumes a weighed graph. If this is undesired, use weight = None.

        A = [nx.dijkstra_path(G, source, target, weight='weight')]
        A_len = [sum([G[A[0][l]][A[0][l + 1]]['weight'] for l in range(len(A[0]) - 1)])]
        B = []

        for i in range(1, k):
            for j in range(0, len(A[-1]) - 1):
                Gcopy = cp.deepcopy(G)
                spurnode = A[-1][j]
                rootpath = A[-1][:j + 1]
                for path in A:
                    if rootpath == path[0:j + 1]:  # and len(path) > j?
                        if Gcopy.has_edge(path[j], path[j + 1]):
                            Gcopy.remove_edge(path[j], path[j + 1])
                        if Gcopy.has_edge(path[j + 1], path[j]):
                            Gcopy.remove_edge(path[j + 1], path[j])
                for n in rootpath:
                    if n != spurnode:
                        Gcopy.remove_node(n)
                try:
                    spurpath = nx.dijkstra_path(Gcopy, spurnode, target, weight='weight')
                    totalpath = rootpath + spurpath[1:]
                    if totalpath not in B:
                        B += [totalpath]
                except nx.NetworkXNoPath:
                    continue
            if len(B) == 0:
                break
            lenB = [sum([G[path[l]][path[l + 1]]['weight'] for l in range(len(path) - 1)]) for path in B]
            B = [p for _, p in sorted(zip(lenB, B))]
            A.append(B[0])
            A_len.append(sorted(lenB)[0])
            B.remove(B[0])

        return A, A_len

    all_od_shortest_paths_dict = {}

    nx_graph_G = nx.from_pandas_edgelist(OW_df, 'init_node', 'term_node',
                                         ['free_flow_time', 'flow', 'cost', 'weight'], create_using=nx.Graph())

    all_od_shortest_paths_dict = {}
    for init_n in OW_demand_df.init_node.unique():
        for term_n in OW_demand_df.term_node.unique():
            if init_n != term_n:
                paths, lengths = k_shortest_paths(
                    nx_graph_G, init_n, term_n, k=ksp, weight='weight')
                all_od_shortest_paths_dict[str(init_n) + '_' + str(term_n)] = paths
    print('all_od_shortest_paths_dict=', all_od_shortest_paths_dict)

    rl_agent_od = list(all_od_shortest_paths_dict.keys())  # rl-agents <--> od-pairs
    one_route_od = []

    print('od number:', len(all_od_shortest_paths_dict))
    print('rl_agent_od number:', len(rl_agent_od))
    print('one_route_od number:', len(one_route_od))


    def edges_from_path(s_path):
        # s_path [1, 3, 4, 5, 6, 2]
        edges = []
        for i in range(len(s_path) - 1):
            o = str(s_path[i])
            d = str(s_path[i + 1])
            edges.append(o + '_' + d)
        return edges  # ['1_3', '3_4', '4_5', '5_6', '6_2']

    route_info = {}
    current_id = 0

    #print(len())
    for od, routes in all_od_shortest_paths_dict.items():
        route_info[od] = {}
        route_info[od]['id_list'] = []
        route_info[od]['route_list'] = []
        route_info[od]['edge_list'] = []
        route_info[od]['link_embedding'] = []  
        route_info[od]['free_flow_tt'] = []
        route_info[od]['n_links'] = []
        for route in routes:
            # route_ids[od]['route_id'] = current_id
            route_info[od]['id_list'].append(current_id)
            route_info[od]['route_list'].append(route)
            route_edges = edges_from_path(route)
            route_info[od]['edge_list'].append(route_edges)
            link_embeddings = OW_df.od.isin(route_edges).values.astype(int)
            route_info[od]['link_embedding'].append(link_embeddings)
            route_info[od]['free_flow_tt'].append(np.dot(link_embeddings, OW_df.free_flow_time))
            route_info[od]['n_links'].append(len(route))
            current_id += 1
    n_routes = current_id
    #route_info['n_routes'] = current_id

    agent_ids = {}  #  {0: '21_1',..., 527: '14_24'}
    for i in range(len(rl_agent_od)):
        agent_ids[i] = rl_agent_od[i]

    print('len(agent_ids):', len(agent_ids))

    # ------------- get agent id map to od-pair -------------
    agent_ids = {}  # {0: '21_1',..., 527: '14_24'}
    for i in range(len(rl_agent_od)):
        agent_ids[i] = rl_agent_od[i]

    print('len(agent_ids):', len(agent_ids))
    print('agent_ids:', agent_ids)

    # ------------- 6. get graph features -------------
    # ------------- get graph features: degree, centrality -------------
    OW_df['edge_idx'] = OW_df.index
    edge_idx_map = dict(zip(OW_df.od, OW_df.edge_idx))
    idx_edge_map = dict(zip(OW_df.edge_idx, OW_df.od))
    net_edges = OW_df.od.values

    edge_as_nodes = []
    for o_edge in net_edges:
        o_edge_o, o_edge_d = o_edge.split('_')
        for d_edge in net_edges:
            d_edge_o, d_edge_d = d_edge.split('_')
            if (o_edge_d == d_edge_o) and (o_edge_o != d_edge_d): 
                new_gat_edge = [o_edge, d_edge]
                if new_gat_edge not in edge_as_nodes:
                    edge_as_nodes.append(new_gat_edge)
    # print(edge_as_nodes)

    df_gat = pd.DataFrame({'init_node': np.array(edge_as_nodes)[:, 0], 'terminate_node': np.array(edge_as_nodes)[:, 1]})
    # print('df_gat.head()=', df_gat.head())
    df_gat['init_idx'] = df_gat['init_node'].map(edge_idx_map)
    df_gat['terminate_idx'] = df_gat['terminate_node'].map(edge_idx_map)
    print('df_gat.head()=', df_gat.head())

    G_gat_two_node = nx.from_pandas_edgelist(df_gat, 'init_idx', 'terminate_idx',
                                             ['init_node', 'terminate_node'], create_using=nx.DiGraph())
    #### get degree
    degree_dict = {}
    for k, v in G_gat_two_node.degree():
        degree_dict[idx_edge_map[k]] = v
    # print('degree_dict=', degree_dict)

    #### get centrality
    betweenness_centrality = nx.betweenness_centrality(G_gat_two_node)
    # print(betweenness_centrality)
    centrality_dict = {}
    for k, v in betweenness_centrality.items():
        centrality_dict[idx_edge_map[k]] = v

    #### add degree_norm and centrality_norm to sioux_falls_df
    OW_df['degree'] = OW_df.od.map(degree_dict)
    OW_df['centrality'] = OW_df.od.map(centrality_dict)
    OW_df['degree_norm'] = (OW_df['degree'] - OW_df['degree'].min()) / (
            OW_df['degree'].max() - OW_df['degree'].min())
    OW_df['centrality_norm'] = (OW_df['centrality'] - OW_df['centrality'].min()) / (
            OW_df['centrality'].max() - OW_df['centrality'].min())

    print('OW_df.columns:', OW_df.columns)

    # ------------- Get config -------------
    OW_config = Config(net_df=OW_df, demand_df=OW_demand_df, all_od_paths=all_od_shortest_paths_dict,
                          action_space=ksp)
    #sf_so_config.so_routes_dist_dict = so_routes_dist_dict
    OW_config.node_xy = node_xy
    OW_config.node_xy_dict = node_xy_dict  # {1: array([0., 1.]),  2: array([0.72972973, 1.]),...}
    # print('node_xy_dict=', node_xy_dict)
    OW_config.route_info = route_info
    OW_config.n_routes = n_routes
    OW_config.od_demand_embeddings = od_demand_embeddings
    OW_config.remove_single_route_od_pairs = remove_single_route_od_pairs
    OW_config.rl_agent_od = rl_agent_od
    OW_config.one_route_od = one_route_od
    OW_config.agent_ids = agent_ids
    #print(agent_ids)
    OW_config.max_env_iter_num = 50
    OW_config.net_name = 'OW'
    OW_config.vehs_per_agent = vehs_per_agent
    print('ksp=', ksp)

    print('OW_demand_df:', OW_demand_df.head(5))

    return OW_config



if __name__ == '__main__':


    args_abs_path = os.path.join(os.getcwd(), 'OW')
    # args = load_SiouxFalls_so_action_set_GAT_config(file_path=args_abs_path)

    args = load_OW_ksp_multi_agent_config(file_path=args_abs_path)

    from STA_Continuous_MultiAgent_Env_V1 import STA_Continuous_MultiAgent_Env_V1
    env = STA_Continuous_MultiAgent_Env_V1(args=args)
    print(env.obs_dim)

    print('!!!!!!!!!!')


