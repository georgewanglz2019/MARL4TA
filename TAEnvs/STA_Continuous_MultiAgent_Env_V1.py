import random
import pandas as pd
import pickle
import numpy as np
import time
import csv
import copy
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import networkx as nx
import copy as cp
import os


class Buffer:
    def __init__(self, size=4, n_rotues=6):
        self.size = size
        self.buffer = [np.zeros(n_rotues, dtype=np.float32) for i in range(size)]

    def add(self, data):
        self.buffer.append(data)
        self.buffer.pop(0)

    def get_history(self):
        return self.buffer


class STA_Continuous_MultiAgent_Env_V1(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, args, render_mode=None):
        super(STA_Continuous_MultiAgent_Env_V1).__init__()

        self.net_name = args.net_name  # siouxfalls or Anaheim
        self.net_df = args.net_df  # sioux_falls_df dataframe data
        self.demand_df = args.demand_df  # sioux_falls_demand_df
        self.full_demand_df = cp.deepcopy(self.demand_df)  #  initial demand
        self.action_dim = args.action_space  # action_dim:  6,  action_space: Box(...)
        self.node_xy = args.node_xy
        self.node_xy_dict = args.node_xy_dict  # {1: array([0., 1.]),  2: array([0.72972973, 1.]),...}
        #self.so_routes_dist_dict = args.so_routes_dist_dict
        self.od_demand_embeddings = args.od_demand_embeddings
        self.routes = args.all_od_paths   # dict['1_2']=[[1, 2], [1, 3, 4, 5, 6, 2]]
        self.min_global_gap = None

        self.route_info = args.route_info
        self.agent_ids = args.agent_ids  # {0: '21_1',  1: '21_4', 2: '21_5',...}
        #print('self.agent_ids=', self.agent_ids)

        self.ue_or_so = args.ue_or_so                    # "ue" or "so" objective
        self.action_smoothing = args.action_smoothing    # float, 0.0: means no action smoothing
        self.max_env_iter_num = args.max_env_iter_num
        self.OW_dynamic_demand = args.OW_dynamic_demand  # 0: fixed demand,  1: dynamic demand

        self.static_obs = self.get_static_obs()  # {'1_2': np.array([...]),...}


        self.rl_agent_od = args.rl_agent_od  # ['1_2',...]
        self.one_route_od = args.one_route_od  # ['1_2',...]
        self.remove_single_route_od_pairs = args.remove_single_route_od_pairs


        self.demand_df['scaling_factor'] = self.demand_df['demand'] / self.demand_df['demand'].sum()

        self.demand_df_dict = {}  # {'2_1':100, ....}
        for k, v in zip(self.demand_df['od'], self.demand_df['demand']):
            self.demand_df_dict[k] = v
        print('total demand:', np.sum(self.demand_df['demand']))
        self.demand_scaling_dict = {}  # {'2_1':100, ....}
        for k, v in zip(self.demand_df['od'], self.demand_df['scaling_factor']):
            self.demand_scaling_dict[k] = v

        # init network flow
        self.net_df = self.reset_net_flow(net_df=self.net_df, demand_dict=self.demand_df_dict)
        #self.net_df['edge_flow'] = 0  # the traffic flow (volume) on each edge
        # update edge state based on 'edge_flow' using bpr function
        self.update_edge_state()

        # save some variables during training
        self.last_n_marg_tt_dict = {key: Buffer(size=4, n_rotues=len(val)) for key, val in self.routes.items()}
        self.last_ass_ratio_dict = {key: np.zeros(len(val), dtype=np.float32) for key, val in self.routes.items()}
        self.last_marg_tt_dict = {key: np.zeros(len(val), dtype=np.float32) for key, val in self.routes.items()}

        self.max_itertation_time = args.max_env_iter_num
        self.iteration_time = 0
        self.env_iter_time = 0
        #self.max_itertation_time = 20
        self.cum_reward = 0
        self.env_reset_time = 0

        self.episode_info = {'episode_global_gap': []}  # for evaluation

        # for light_mappo code:
        obs_space = len(self.get_obs()[0])
        self.num_agent = len(self.agent_ids)  # set the number of agents
        #print('self.agent_num=', self.agent_num ) 528
        self.obs_dim = obs_space  # set the observation dimension of agents
        #self.action_dim = self.action_dim  # set the action dimension of agents, here set to a five-dimensional
        self.env_reset_time = 0

        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim,), dtype=np.float32)

        print('self.action_space:', self.action_space)
        print('self.observation_space:', self.observation_space)

        # save experiences
        self.TTTs = []  # save all total travel times during running
        self.save_decison_details = False

        self.last_local_gap = None
        self.last_global_gap = None
        self.global_gap_list = []
        self.last_global_gap_list = None  # not reset to None in reset()
        self.last_demand_df = None  # not reset to None in reset()
        self.last_net_df = None  # not reset to None in reset()


    def reset_net_flow(self, net_df, demand_dict):
        net_df['edge_flow'] = 0

        if self.remove_single_route_od_pairs:
            # for od-pair with only one route in routes set
            # one_route_demand = 0
            # print('self.one_route_od=', self.one_route_od)
            for od in self.one_route_od:
                link_emb = self.route_info[od]['link_embedding'][0]
                demand = self.demand_df_dict[od]  # self.demand_df_dict = {'2_1':100, ....}
                net_df['edge_flow'] += link_emb * demand
                # one_route_demand += demand
            # print('one_route_demand:', one_route_demand)
        return net_df


    def update_edge_state(self):
        # update edge tt, additional tt, marginal tt, MTT_derivative, occupancy and over_occupancy based on 'edge_flow'
        self.net_df = self.update_net_df(net_df=self.net_df)


    def update_net_df(self, net_df):
        edge_flow = net_df['edge_flow'].values
        capacity = net_df['capacity'].values
        free_flow_time = net_df['free_flow_time'].values
        b = net_df['b'].values
        power = net_df['power'].values

        occupancy = edge_flow / capacity
        occupancy_power = (edge_flow / capacity) ** power
        occupancy_power_minus_1 = (edge_flow / capacity) ** (power - 1)

        net_df['edge_tt'] = free_flow_time * (1 + b * occupancy_power)
        net_df['edge_add_tt'] = free_flow_time * b * power * occupancy_power_minus_1 / capacity
        net_df['edge_marginal_tt'] = free_flow_time * (1 + b * (power + 1) * occupancy_power)
        net_df['edge_MTT_derivative'] = free_flow_time * b * power * (power + 1) * occupancy_power_minus_1 / capacity
        net_df['occupancy'] = occupancy
        net_df['over_occupancy'] = (occupancy > 0.95).astype(int)
        return net_df


    def reset(self, seed=None, options=None):
        # need the following line to seed self.np_random
        super().reset(seed=seed)
        self.seed(seed=seed)

        self.demand_df = cp.deepcopy(self.full_demand_df)  # sioux_falls_demand_df
        # self.demand_df['ratio'] = np.ones(len(self.demand_df))  # 1
        if self.net_name == 'siouxfalls':  # siouxfalls or Anaheim
            self.demand_df['ratio'] = (np.random.random(len(self.demand_df)) + 1) / 2   # 0.5 ~ 1
        elif self.net_name == 'OW':
            if self.OW_dynamic_demand == 1:
                self.demand_df['ratio'] = (np.random.random(len(self.demand_df)) + 1) / 2  # 0.5 ~ 1
            else:
                self.demand_df['ratio'] = 1  # 1
        elif self.net_name == 'Anaheim':
            self.demand_df['ratio'] = np.random.random(len(self.demand_df)) + 0.5  # 0.5 ~ 1.5
        else:
            raise ValueError(f'Net name = {self.net_name} not recognized')


  
        self.demand_df['demand'] = self.demand_df['ratio'] * self.demand_df['demand']
        self.demand_df['scaling_factor'] = self.demand_df['demand'] / self.demand_df['demand'].sum()
        min_scal_f, max_scal_f = self.demand_df['scaling_factor'].min(), self.demand_df['scaling_factor'].max()
        self.demand_df['scaling_factor'] = (self.demand_df['scaling_factor'] - min_scal_f)/(max_scal_f - min_scal_f)
        self.demand_df_dict = {}  # {'2_1':100, ....}
        for k, v in zip(self.demand_df['od'], self.demand_df['demand']):
            self.demand_df_dict[k] = v
        print('total demand:', np.sum(self.demand_df['demand']))
        self.demand_scaling_dict = {}  # {'2_1':0.5, ....}
        for k, v in zip(self.demand_df['od'], self.demand_df['scaling_factor']):
            self.demand_scaling_dict[k] = v

        self.net_df = self.reset_net_flow(net_df=self.net_df, demand_dict=self.demand_df_dict)
        # update edge state based on 'edge_flow' using bpr function
        self.update_edge_state()

        self.iteration_time = 0
        self.cum_reward = 0
        self.env_reset_time += 1
        self.episode_info = {'episode_global_gap': []}  # for evaluation

        # get_obs()
        obs = self.get_obs()

        self.last_local_gap = None
        self.last_global_gap = None
        self.global_gap_list = []



        return obs  # for light_ppo repo: env_core

    def get_static_obs(self):
        static_obs = {}  # {'1_2': np.array([...]),...}
        for agent_id in list(self.agent_ids.keys()):  # 0 --> 528
            od = self.agent_ids[agent_id]  # '1_2'
            #print('agent_id=', agent_id, 'od=', od)
            r_info = self.route_info[od]   # '2_3' --> {'id_list': [0, 1],  'route_list': [[2,3],...], 'link_embedding':...}
            n_links = len(r_info['id_list'])

            ########## get static obs/features
            o_xy = self.node_xy_dict[int(od.split('_')[0])]  # array([0.21621622, 0.30434783])
            d_xy = self.node_xy_dict[int(od.split('_')[1])]

            if self.net_name == 'siouxfalls':  # siouxfalls or Anaheim
                od_demand_embeddings = self.od_demand_embeddings[od] / 4000  #  array([ 100.,-100.,0.,0., ... ,0])
                r_ids = np.array(r_info['id_list']) / 3000
                free_flow_tt = np.array(
                    [np.dot(arr, self.net_df['free_flow_time']) for arr in r_info['link_embedding']],
                    dtype=np.float32) / 35
                n_links_array = np.array([n_links]) / 10

            elif self.net_name == 'OW':
                od_demand_embeddings = self.od_demand_embeddings[od] / 400  # array([ 100.,-100.,0.,0., ... ,0])
                r_ids = np.array(r_info['id_list']) / 20
                free_flow_tt = np.array(
                    [np.dot(arr, self.net_df['free_flow_time']) for arr in r_info['link_embedding']],
                    dtype=np.float32) / 20
                n_links_array = np.array([n_links]) / 5

            elif self.net_name == 'Anaheim':
                od_demand_embeddings = self.od_demand_embeddings[od] / 400  # array([ 100.,-100.,0.,0., ... ,0])
                r_ids = np.array(r_info['id_list']) / 4000
                free_flow_tt = np.array(
                    [np.dot(arr, self.net_df['free_flow_time']) for arr in r_info['link_embedding']],
                    dtype=np.float32) / 20
                n_links_array = np.array([n_links]) / 20
            else:
                raise ValueError(f'Net name = {self.net_name} not recognized')


            degree_norm = np.array(  
                [np.dot(arr / r_info['n_links'][i], self.net_df['degree_norm']) for i, arr in
                 enumerate(r_info['link_embedding'])],
                dtype=np.float32)
            centrality_norm= np.array(  
                [np.dot(arr / r_info['n_links'][i], self.net_df['centrality_norm']) for i, arr in
                 enumerate(r_info['link_embedding'])],
                dtype=np.float32)

            static_obs_i = np.concatenate(
                [o_xy, d_xy, od_demand_embeddings, r_ids, free_flow_tt, n_links_array, degree_norm, centrality_norm])

            # static_obs_i = np.concatenate(
            #     [o_xy, d_xy, od_demand_embeddings, r_ids, n_links_array])

            static_obs[od] = static_obs_i

        return static_obs  # {'1_2': np.array([...]),...}


    def get_obs(self):
        obs = []

        edge_flow = self.net_df['edge_tt'].values
        edge_tt = self.net_df['edge_tt'].values
        edge_occupancy = self.net_df['occupancy'].values
        edge_over_occupancy = self.net_df['over_occupancy'].values
        edge_marginal_tt = self.net_df['edge_marginal_tt'].values
        edge_MTT_derivative = self.net_df['edge_MTT_derivative'].values

        for agent_id in list(self.agent_ids.keys()):  # 0 --> 528
            od = self.agent_ids[agent_id]  # '1_2'
            r_info = self.route_info[od]   # '2_3' --> {'id_list': [0, 1],  'route_list': [[2,3],...], 'link_embedding':...}

            ########## get dynamic obs/features
            if self.net_name == 'siouxfalls':  # siouxfalls or Anaheim
                marginal_tt = np.array(
                    [np.dot(arr, edge_marginal_tt) for arr in r_info['link_embedding']],
                    dtype=np.float32) / (20 * 6)
                marginal_tt_gap = marginal_tt - marginal_tt.min()
                #self.last_marg_tt_dict[od] = marginal_tt  # update last marginal travel time
                tt = np.array(   
                    [np.dot(arr/r_info['n_links'][i], edge_tt) for i, arr in enumerate(r_info['link_embedding'])],
                    dtype=np.float32) / 15
                flow = np.array(
                    [np.dot(arr/r_info['n_links'][i], edge_flow) for i, arr in enumerate(r_info['link_embedding'])],
                    dtype=np.float32) / 20000
                vc = np.array(
                    [np.dot(arr/r_info['n_links'][i], edge_occupancy) for i, arr in enumerate(r_info['link_embedding'])],
                    dtype=np.float32)   
                over_vc = np.array(
                    [np.dot(arr, edge_over_occupancy) for i, arr in enumerate(r_info['link_embedding'])],
                    dtype=np.float32) / 5
                MTT_derivative = np.array(
                    [np.dot(arr, edge_MTT_derivative) for i, arr in enumerate(r_info['link_embedding'])],
                    dtype=np.float32) / 3.4e-3

            elif self.net_name == 'OW':
                marginal_tt = np.array(
                    [np.dot(arr, edge_marginal_tt) for arr in r_info['link_embedding']],
                    dtype=np.float32) / (8 * 5)
                marginal_tt_gap = marginal_tt - marginal_tt.min()
                # self.last_marg_tt_dict[od] = marginal_tt  # update last marginal travel time
                tt = np.array(  
                    [np.dot(arr / r_info['n_links'][i], edge_tt) for i, arr in enumerate(r_info['link_embedding'])],
                    dtype=np.float32) / 8
                flow = np.array(
                    [np.dot(arr / r_info['n_links'][i], edge_flow) for i, arr in enumerate(r_info['link_embedding'])],
                    dtype=np.float32) / 150
                vc = np.array(
                    [np.dot(arr / r_info['n_links'][i], edge_occupancy) for i, arr in
                     enumerate(r_info['link_embedding'])],
                    dtype=np.float32) 
                over_vc = np.array(
                    [np.dot(arr, edge_over_occupancy) for i, arr in enumerate(r_info['link_embedding'])],
                    dtype=np.float32) / 3
                MTT_derivative = np.array(
                    [np.dot(arr, edge_MTT_derivative) for i, arr in enumerate(r_info['link_embedding'])],
                    dtype=np.float32) / 5e-3

            elif self.net_name == 'Anaheim':
                marginal_tt = np.array(
                    [np.dot(arr, edge_marginal_tt) for arr in r_info['link_embedding']],
                    dtype=np.float32) / (1.5 * 20)
                marginal_tt_gap = marginal_tt - marginal_tt.min()
                # self.last_marg_tt_dict[od] = marginal_tt  # update last marginal travel time
                tt = np.array(  
                    [np.dot(arr / r_info['n_links'][i], edge_tt) for i, arr in enumerate(r_info['link_embedding'])],
                    dtype=np.float32) / 1.2
                flow = np.array(
                    [np.dot(arr / r_info['n_links'][i], edge_flow) for i, arr in enumerate(r_info['link_embedding'])],
                    dtype=np.float32) / 3200
                vc = np.array(
                    [np.dot(arr / r_info['n_links'][i], edge_occupancy) for i, arr in
                     enumerate(r_info['link_embedding'])],
                    dtype=np.float32)  
                over_vc = np.array(
                    [np.dot(arr, edge_over_occupancy) for i, arr in enumerate(r_info['link_embedding'])],
                    dtype=np.float32) / 5
                MTT_derivative = np.array(
                    [np.dot(arr, edge_MTT_derivative) for i, arr in enumerate(r_info['link_embedding'])],
                    dtype=np.float32) / 3.4e-3

            else:
                raise ValueError(f'Net name = {self.net_name} not recognized')


            last_4_marg_tt = np.concatenate(self.last_n_marg_tt_dict[od].get_history(), dtype=np.float32)
            last_ass_ratio = self.last_ass_ratio_dict[od]

            dynamic_obs_i = np.concatenate(
                [marginal_tt, marginal_tt_gap, tt, flow, vc, over_vc, MTT_derivative, last_4_marg_tt, last_ass_ratio], dtype=np.float32)
            #dynamic_obs_i_clip = np.clip(dynamic_obs_i, a_min=-1.1, a_max=10)
            if self.net_name == 'siouxfalls':  # siouxfalls or Anaheim
                dynamic_obs_i_clip = np.clip(dynamic_obs_i, a_min=-10, a_max=10)
            elif self.net_name == 'Anaheim' or self.net_name == 'OW':
                dynamic_obs_i_clip = np.clip(dynamic_obs_i, a_min=-15, a_max=15)
            else:
                raise ValueError(f'Net name = {self.net_name} not recognized')


            ########## get static obs/features
            static_obs_i = self.static_obs[od]

            if self.net_name == 'siouxfalls':  # siouxfalls or Anaheim
                static_obs_i_clip = np.clip(static_obs_i, a_min=-10, a_max=10)
            elif self.net_name == 'Anaheim'or self.net_name == 'OW':
                static_obs_i_clip = np.clip(static_obs_i, a_min=-15, a_max=15)
            else:
                raise ValueError(f'Net name = {self.net_name} not recognized')

            obs_i = np.concatenate([dynamic_obs_i_clip, static_obs_i_clip], dtype=np.float32)

            obs.append(obs_i)
        return obs

    def get_state(self, local_gaps_dict, actions):
        # global state

        local_gaps = []
        for agent_id in range(len(self.agent_ids.keys())):  # 0 --> 528
            od = self.agent_ids[agent_id]  # '1_2'
            local_gaps.append(local_gaps_dict[od])

        local_gaps_arr = np.array(local_gaps, dtype=np.float32)
        ass_ratio_arr = np.concatenate(actions, dtype=np.float32)
        #self.net_df['occupancy'].values

        state = np.concatenate([
                                local_gaps_arr, ass_ratio_arr,
                                self.net_df['edge_marginal_tt'].values / 20,
                                self.net_df['edge_tt'].values / 15,
                                self.net_df['flow'].values / 20000,
                                self.net_df['occupancy'].values,
                                self.net_df['over_occupancy'].values,
                                self.demand_df.demand.values / 4000
                               ], dtype=np.float32)
        state_clip = np.clip(state, a_min=-1.1, a_max=10)

        return state_clip


    def render(self):
        return True


    def edges_from_path(self, s_path):
        # s_path [1, 3, 4, 5, 6, 2]
        edges = []
        for i in range(len(s_path) - 1):
            o = str(s_path[i])
            d = str(s_path[i + 1])
            edges.append(o + '_' + d)
        return edges  # ['1_3', '3_4', '4_5', '5_6', '6_2']

    def update_env(self, actions, objective='so'):
        # objective: ue or so
        # update env with acitons
        # self.agent_ids = {0: '21_1',  1: '21_4', 2: '21_5',...}

        # init network flow
        # self.net_df['edge_flow'] = 0  # the traffic flow (volume) on each edge
        self.net_df = self.reset_net_flow(net_df=self.net_df, demand_dict=self.demand_df_dict)
        # update edge state based on 'edge_flow' using bpr function
        #self.update_edge_state()

        ass_edge_flow = np.zeros(len(self.net_df['edge_flow']))

        # rl_ass_demand = 0
        for agent_id, action in enumerate(actions):
            od = self.agent_ids[agent_id]  # '1_2'
            r_info = self.route_info[od]  # '2_3' --> {'id_list': [0, 1],  'route_list': [[2,3],...],...}

            demand = self.demand_df_dict[od]  # self.demand_df_dict = {'2_1':100, ....}
            # rl_ass_demand += demand

            for k, link_emb in enumerate(r_info['link_embedding']):  # update network flow
                ass_edge_flow += demand * action[k] * link_emb

        self.net_df['edge_flow'] += ass_edge_flow

        # update edge state based on 'edge_flow' using bpr function
        self.update_edge_state()
        if objective == 'so':
            self.net_df['weight'] = self.net_df['edge_marginal_tt']  # for calculating shortest path
        else:
            self.net_df['weight'] = self.net_df['edge_tt']  # for calculating shortest path

        #### update some useful variables
        for agent_id, action in enumerate(actions):
            od = self.agent_ids[agent_id]  # '1_2'
            r_info = self.route_info[od]  # '2_3' --> {'id_list': [0, 1],  'route_list': [[2,3],...],...}
            self.last_n_marg_tt_dict[od].add(action)
            self.last_ass_ratio_dict[od] = action
            if objective == 'so':
                if self.net_name == 'siouxfalls':  # siouxfalls or Anaheim
                    self.last_marg_tt_dict[od] = np.array(
                        [np.dot(arr, self.net_df['edge_marginal_tt']) for arr in r_info['link_embedding']],
                        dtype=np.float32) / (20 * 6)
                elif self.net_name == 'Anaheim':
                    self.last_marg_tt_dict[od] = np.array(
                        [np.dot(arr, self.net_df['edge_marginal_tt']) for arr in r_info['link_embedding']],
                        dtype=np.float32) / (1.5 * 20)
                elif self.net_name == 'OW':
                    self.last_marg_tt_dict[od] = np.array(
                        [np.dot(arr, self.net_df['edge_marginal_tt']) for arr in r_info['link_embedding']],
                        dtype=np.float32) / (8 * 5)
                else:
                    raise ValueError(f'Net name = {self.net_name} not recognized')

            else:  # ue
                if self.net_name == 'siouxfalls':  # siouxfalls or Anaheim
                    self.last_marg_tt_dict[od] = np.array(
                        [np.dot(arr, self.net_df['edge_tt']) for arr in r_info['link_embedding']],
                        dtype=np.float32) / (15 * 6)
                elif self.net_name == 'Anaheim':
                    self.last_marg_tt_dict[od] = np.array(
                        [np.dot(arr, self.net_df['edge_tt']) for arr in r_info['link_embedding']],
                        dtype=np.float32) / (1.5 * 20)
                elif self.net_name == 'OW':
                    self.last_marg_tt_dict[od] = np.array(
                        [np.dot(arr, self.net_df['edge_tt']) for arr in r_info['link_embedding']],
                        dtype=np.float32) / (8 * 5)
                else:
                    raise ValueError(f'Net name = {self.net_name} not recognized')

        ##### get relative gap for each od/agent
        local_gaps_dict = {}  # {'1_2':0.2,...}
        SP_COST = 0   #  shortest path cost
        if objective == 'so':
            TS_COST = np.dot(self.net_df['edge_marginal_tt'], self.net_df['edge_flow'])  # current total assignment cost
        else:  # ue
            TS_COST = np.dot(self.net_df['edge_tt'], self.net_df['edge_flow'])  # current total assignment cost

        edge_marginal_tt = self.net_df['edge_marginal_tt'].values
        edge_tt = self.net_df['edge_tt'].values

        if objective == 'so':
            net_graph_G = nx.from_pandas_edgelist(self.net_df, 'init_node', 'term_node', ['edge_marginal_tt'],
                                                  create_using=nx.DiGraph())
        else:  # ue
            net_graph_G = nx.from_pandas_edgelist(self.net_df, 'init_node', 'term_node', ['edge_tt'],
                                                  create_using=nx.DiGraph())

        for agent_id, action in enumerate(actions):
            #print('agent_id=', agent_id)
            od = self.agent_ids[agent_id]  # '1_2'
            demand = self.demand_df_dict[od]  # self.demand_df_dict = {'2_1':100, ....}
            r_info = self.route_info[od]  # '2_3' --> {'id_list': [0, 1],  'route_list': [[2,3],...],...}
            #print('path choices:', r_info['route_list'])


            o_n, d_n = int(od.split('_')[0]), int(od.split('_')[1])
            if objective == 'so':
                sp = nx.shortest_path(G=net_graph_G, source=o_n, target=d_n, weight='edge_marginal_tt')  # e.g., [1, 2]
            else:  # ue
                sp = nx.shortest_path(G=net_graph_G, source=o_n, target=d_n, weight='edge_tt')  # e.g., [1, 2]
            #sp_length = nx.shortest_path_length(G=net_graph_G, source=o_n, target=d_n, weight='edge_marginal_tt')
            sp_edges = self.edges_from_path(s_path=sp)  # # ['1_3', '3_4', '4_5', '5_6', '6_2']
            # #print('shortest path:', sp, 'sp_length=', sp_length)
            sp_link_emb = self.net_df.od.isin(sp_edges).values.astype(int)  # array([0, 0, 0, 0, 1, 1...])
            if objective == 'so':
                sp_length = round(np.dot(sp_link_emb, edge_marginal_tt), 9)
                sp_c = round(np.dot(demand * sp_link_emb, edge_marginal_tt), 9)  # shortest path cost
            else:
                sp_length = round(np.dot(sp_link_emb, edge_tt), 9)
                sp_c = round(np.dot(demand * sp_link_emb, edge_tt), 9)  # shortest path cost

            SP_COST += sp_c

            #### get current assignment cost of each agent/od
            current_cost = 0
            ass_demand = 0
            for k, link_emb in enumerate(r_info['link_embedding']):  # update network flow
                if objective == 'so':
                    current_cost += np.dot(demand * action[k] * link_emb, edge_marginal_tt)
                    ass_demand += demand * action[k]
                else:  # ue
                    current_cost += np.dot(demand * action[k] * link_emb, edge_tt)
                    ass_demand += demand * action[k]

            local_gap = (current_cost / sp_c) - 1 
            if local_gap < 0 and local_gap >= -2e-7:
                local_gap = 0
            if local_gap <= -2e-7:
                print('local_gap <= -2e-7 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('local_gap=', local_gap,'< 0')
                print('ass_demand:', ass_demand, 'demand:', demand)
                print('current_cost=', current_cost, 'sp_c:', sp_c, 'local_gap:', local_gap)
                print('sp_length=', sp_length, 'sp=', sp)
            local_gaps_dict[od] = local_gap
            #local_cost_dict[od] = local_cost

        if self.remove_single_route_od_pairs:
            for od in self.one_route_od:
                demand = self.demand_df_dict[od]  # self.demand_df_dict = {'2_1':100, ....}
                o_n, d_n = int(od.split('_')[0]), int(od.split('_')[1])
                if objective == 'so':
                    sp = nx.shortest_path(G=net_graph_G, source=o_n, target=d_n, weight='edge_marginal_tt')  # e.g., [1, 2]
                else:  # ue
                    sp = nx.shortest_path(G=net_graph_G, source=o_n, target=d_n, weight='edge_tt')
                # sp_length = nx.shortest_path_length(G=net_graph_G, source=o_n, target=d_n, weight='edge_marginal_tt')
                sp_edges = self.edges_from_path(s_path=sp)  # # ['1_3', '3_4', '4_5', '5_6', '6_2']
                # #print('shortest path:', sp, 'sp_length=', sp_length)
                sp_link_emb = self.net_df.od.isin(sp_edges).values.astype(int)  # array([0, 0, 0, 0, 1, 1...])
                if objective == 'so':
                    sp_length = round(np.dot(sp_link_emb, edge_marginal_tt), 9)
                    sp_c = round(np.dot(demand * sp_link_emb, edge_marginal_tt), 9)  # shortest path cost
                else:  # ue
                    sp_length = round(np.dot(sp_link_emb, edge_tt), 9)
                    sp_c = round(np.dot(demand * sp_link_emb, edge_tt), 9)  # shortest path cost
                SP_COST += sp_c

        global_gap = (TS_COST / SP_COST) - 1

        print('global_gap:', global_gap)

        return local_gaps_dict, global_gap


    def get_total_tt(self):
        # get total travel times after an episode
        total_tt = np.dot(self.net_df.edge_flow.values, self.net_df.edge_tt.values)
        return total_tt

    def seed(self, seed):
        # seed
        random.seed(seed)
        np.random.seed(seed)

    def close(self):
        pass

    def adjust_action(self, actions, mode='softmax'):
        # actions: action by the RL
        # mode: softmax of clip

        if mode == 'clip':
            def adjust_actions(acts):
                adjusted_actions = []
                for action in acts:
                    adjusted_action = np.zeros_like(action)
                    cumulative_sum = 0

                    for i in range(len(action)):
                        action = np.clip(action, 0, 1)

                        if cumulative_sum + action[i] <= 1:
                            adjusted_action[i] = action[i]
                            cumulative_sum += action[i]
                        else:
                            adjusted_action[i] = 1 - cumulative_sum
                            break

                    if np.sum(adjusted_action) < 1:
                        adjusted_action[-1] += 1 - cumulative_sum

                    adjusted_actions.append(adjusted_action)

            actions = adjust_actions(acts=actions)

        elif mode == 'softmax':
            actions = [self.softmax(logits=action) for action in actions]
            # print('after apply threshold actions[0]=', actions[0])

        return actions

    def softmax(self, logits, clip_threshold=0.003):
        """
        Compute the softmax of a vector of logits and clip small probabilities,
        then renormalize to ensure the sum of probabfilities is 1.

        Parameters:
        logits (numpy array): A numpy array of logits.
        clip_threshold (float): A threshold below which probabilities are set to zero.

        Returns:
        numpy array: A numpy array of renormalized softmax probabilities.
        """
        exp_logits = np.exp(
            logits - np.max(logits))  # Stability improvement: subtract max logit for numerical stability
        softmax_probs = exp_logits / np.sum(exp_logits)

        # Clip small probabilities to zero
        softmax_probs = np.where(softmax_probs < clip_threshold, 0, softmax_probs)

        # Renormalize probabilities to ensure they sum to 1
        softmax_probs /= np.sum(softmax_probs)

        return softmax_probs

    def softmax_inverse(self, probs, epsilon=0.003):
        """
        Given renormalized softmax probabilities, compute the corresponding logits.

        Parameters:
        probs (numpy array): A numpy array of renormalized softmax probabilities.
        epsilon (float): A small value to replace zero probabilities to avoid log(0).
        small_value (float): A small negative value to assign to logit of zero probabilities.

        Returns:
        numpy array: A numpy array of logits, normalized such that the sum of the logits is zero.
        """
        # Ensure the probabilities are a numpy array
        probs = np.array(probs)

        # Replace zeros with epsilon to avoid log(0)
        probs = np.clip(probs, epsilon, 1.0)

        # Log probabilities
        log_probs = np.log(probs)

        # Handle logits: assign a small negative value to zero probabilities to avoid -inf
        # log_probs[probs == epsilon] = small_value

        # Choose a reference logit, here use the mean to ensure the sum of logits is zero
        mean_log_prob = np.mean(log_probs)

        # Calculate the logits relative to the mean log probability
        logits = log_probs - mean_log_prob

        return logits


    def step(self, actions):
        # for continuous Env: action should be like np.array([0.5, 0.2, 0.3])
        # for multi_agent_env: actions = [np.array([0.5, 0.2, 0.3]), np.array([0.1, 0.2, 0.7]),...]
        self.iteration_time += 1
        self.env_iter_time += 1    # not reset in reset()

        if self.iteration_time >= self.max_itertation_time:
            print('------ Env iteration time =', self.env_iter_time)

        print('actions[0]=', actions[0])

        def apply_threshold(probs, threshold=0.0001):
            probs[probs < threshold] = 0  
            probs /= probs.sum() 
            return probs
        actions = copy.deepcopy(actions)
        actions = [apply_threshold(probs=action, threshold=self.action_smoothing) for action in actions]
        print('after apply threshold actions[0]=', actions[0])

        # local_gaps_dict, global_gap = self.update_env(actions=actions, objective='ue')  # return local_gaps_dict, global_gap
        local_gaps_dict, global_gap = self.update_env(actions=actions,
                                                      objective=self.ue_or_so)  # return local_gaps_dict, global_gap
        self.episode_info['episode_global_gap'].append(global_gap)
        self.last_global_gap_list = copy.copy(self.episode_info['episode_global_gap'])
        min_global_gap = min(self.episode_info['episode_global_gap'])
        self.min_global_gap = min_global_gap
        print("Minimun gap so far=", min_global_gap, ' its iteration num=', self.episode_info['episode_global_gap'].index(min_global_gap))

        if self.iteration_time <= 1:
            self.last_demand_df = copy.copy(self.demand_df)
            self.last_net_df = copy.copy(self.net_df)

        # gap_threshold = 1e-4

        # get rewards
        local_rewards = []
        for agent_id, action in enumerate(actions):
            od = self.agent_ids[agent_id]  # '1_2'
            if self.iteration_time <= 1:
                #local_rewards.append([-local_gaps_dict[od]])
                local_r = - local_gaps_dict[od]
                global_r = - global_gap
                local_rewards.append([0.8 * local_r + 0.2 * global_r]) 
                # local_rewards.append([local_r])
                # local_rewards.append([200 * self.demand_scaling_dict[od] * (8 * local_r + 2 * global_r)])

            else:
                local_r = self.last_local_gap[od] - local_gaps_dict[od]
                global_r = self.last_global_gap - global_gap
                # if global_gap <= gap_threshold:
                #     global_r = 10  # converge --> give positive reward!
                local_rewards.append([0.8 * local_r + 0.2 * global_r]) 
                # local_rewards.append([200 * self.demand_scaling_dict[od] * (8*local_r + 2*global_r)])
                # local_rewards.append([local_r]) 

        self.last_local_gap = local_gaps_dict
        self.last_global_gap = global_gap
        global_reward = -global_gap

        total_tt = self.get_total_tt()
        print('------ self.iteration_time=', self.iteration_time, ' gap=', global_gap, ' total_tt=', total_tt)
        print('------ local reward mean:', np.mean(local_rewards), '  std:', np.std(local_rewards), 'min:', min(local_rewards), 'max:', max(local_rewards))
        #print('------ local_gaps_dict[od]:', local_gaps_dict)
        local_gaps_list = list(local_gaps_dict.values())
        print('------ local_gaps_dict:', local_gaps_list[:30])
        print('------ local_gaps_dict mean=', np.mean(local_gaps_list), ', std=', np.std(local_gaps_list))
        print('total_tt=', total_tt)


        self.cum_reward += -global_gap

        obs = self.get_obs()
        state = self.get_state(local_gaps_dict=local_gaps_dict, actions=actions)

        dones = []

        if self.iteration_time >= self.max_itertation_time:
            for i in range(self.num_agent):
                dones.append(True)
            self.TTTs.append(total_tt)

        else:  # not converge
            for i in range(self.num_agent):
                dones.append(False)

        infos = [np.zeros(self.action_dim) for _ in range(len(actions))]


        return [obs, local_rewards, dones, infos]  # for light_ppo repo: env_core

    def get_best_tt(self):
        return min(self.TTTs)

    def get_odpairs_shortest_path(self, net_df, od_pairs, weight, get_sp_cost=True):
        """
        get the shortest path of given od-pairs and net-dataframe file.
        :param net_df: (pd.DataFrame) netwrok data
        :param od_pairs: (list) od pairs list. e.g., ['1_2', '8_11',...]
        :param weight: weight/cost attribute of the edges. e.g., 'edge_marginal_tt' or 'edge_tt'
        :param get_sp_cost: whether get the cost of the shortest path

        :return sp_dict: (dict) shortest path dict. e.g., {'1_2':[1,2],...}
        :return sp_cost_dict: (dict) cost of the shortest path. e.g., {'1_2':10.}
        """
        nx_G = nx.from_pandas_edgelist(net_df, 'init_node', 'term_node', [weight],
                                             create_using=nx.DiGraph())
        sp_dict = {}
        sp_cost_dict = {}
        for od_pair in od_pairs:
            origin, destination = int(od_pair.split('_')[0]), int(od_pair.split('_')[1])

            sp = nx.shortest_path(nx_G, source=origin, target=destination, weight=weight)
            sp_dict[od_pair] = sp
            if get_sp_cost:
                sp_length = nx.shortest_path_length(nx_G, source=origin, target=destination, weight=weight)
                sp_cost_dict[od_pair] = sp_length

        return sp_dict, sp_cost_dict

    def optimize_one_od_using_GP(self, assign_info, net_df, od_pair, weight):
        """
        get optimized path flow/distribution for one specific od-pair using path-based method (gradient projection, GP)
        :param assign_info: (dict) current assignment solution. e.g., {'route_list':[[1,2],...], 'assign_ratio': np.array([0.5, ...]),
                            'assign_demand': np.array([500, ...]),'link_embedding': [array([0, 0, 1, 0,...]),...]}
        :param net_df: (pd.DataFrame) netwrok data
        :param od_pair: (list) one od pair list. e.g., '1_2'
        :param weight: weight/cost attribute of the edges. e.g., 'edge_marginal_tt' or 'edge_tt'
        :return sp_dict: (dict) shortest path dict. e.g., {'1_2':[1,2],...}
        """
        sp_dict, sp_cost_dict = self.get_odpairs_shortest_path(net_df=net_df, od_pairs=[od_pair],
                                                               weight=weight)  # {'1_2':[1,2],.}
        sp = sp_dict[od_pair]  # [1, 2]
        sp_edges = self.edges_from_path(s_path=sp)  # ['1_3', '3_4', '4_5', '5_6', '6_2']
        sp_link_embedding = self.net_df.od.isin(sp_edges).values.astype(int)
        sp_cost = sp_cost_dict[od_pair]

        # sp_assigned_demand = 0
        optimized_route_list = [] 
        optimized_assign_demand = []
        optimized_link_embedding = []

        sp_in_assigned_route = sp in assign_info['route_list']
        # case1: shortest path is in assigned route list
        if sp_in_assigned_route:
            sp_index = assign_info['route_list'].index(sp)
            # sp_ass_ratio = assign_info['assign_ratio'][sp_index]
            sp_assigned_demand = assign_info['assign_demand'][sp_index]
        # case2: shortest path is not in assigned route list
        else:
            sp_assigned_demand = 0
            sp_index = len(assign_info['route_list'])

        edge_flow_shift = np.zeros(len(sp_link_embedding))  # save edge flow shift for network loading
        # update for each route
        for r_i, (route, r_i_demand, r_i_link_embedding) in enumerate(zip(assign_info['route_list'],
                                                                          assign_info['assign_demand'],
                                                                          assign_info['link_embedding'])):
            if r_i_demand > 0 and r_i != sp_index:
                r_i_cost = np.dot(r_i_link_embedding, net_df['edge_marginal_tt'])
                assert round(r_i_cost, 6) >= round(sp_cost, 6)  # make sure that non-basic path has bigger cost

                # the links which are used by either π or ˆ π, but not both.
                derivative_links = np.where(sp_link_embedding != r_i_link_embedding, 1, 0)
                derivative_sum = np.dot(derivative_links, net_df['edge_MTT_derivative'])
                theory_shift_demand = (round(r_i_cost, 6) - round(sp_cost,
                                                                  6)) / derivative_sum  # use round(,6) to make sure sometimes r_i_cost=6.0000321, sp_cost=6.000032
                shift_demand = min(theory_shift_demand, r_i_demand)

                # adjust demand to sp
                sp_assigned_demand += shift_demand
                r_i_demand -= shift_demand

                edge_flow_shift += shift_demand * (sp_link_embedding - r_i_link_embedding)

            optimized_route_list.append(route)
            optimized_assign_demand.append(r_i_demand)
            optimized_link_embedding.append(r_i_link_embedding)

        if sp_in_assigned_route:  # shortest path is in assigned route list
            optimized_assign_demand[sp_index] = sp_assigned_demand  # optimized sp assigned demand
        else:  
            optimized_route_list.append(sp)
            optimized_assign_demand.append(sp_assigned_demand)
            optimized_link_embedding.append(sp_link_embedding)

        optimized_assign_demand = np.array(optimized_assign_demand)
        total_ass_demand = np.sum(optimized_assign_demand)
        if total_ass_demand == 0:
            raise ValueError("The sum of the array elements is zero, cannot divide by zero.")
        optimized_assign_ratio = optimized_assign_demand / total_ass_demand

        optimized_assign_info = {
            'route_list': optimized_route_list,
            'assign_ratio': optimized_assign_ratio,
            'assign_demand': optimized_assign_demand,
            'link_embedding': optimized_link_embedding
        }
        return optimized_assign_info, edge_flow_shift

    def optimize_one_od_using_GP_with_fixed_routes(self, assign_info, net_df, od_pair, weight):
        """
        get optimized path flow/distribution for one specific od-pair using path-based method (gradient projection, GP),
        however in this func, have a fixed route set for each od-pair

        :param assign_info: (dict) current assignment solution. e.g., {'route_list':[[1,2],...], 'assign_ratio': np.array([0.5, ...]),
                            'assign_demand': np.array([500, ...]),'link_embedding': [array([0, 0, 1, 0,...]),...]}
        :param net_df: (pd.DataFrame) netwrok data
        :param od_pair: (list) one od pair list. e.g., '1_2'
        :param weight: weight/cost attribute of the edges. e.g., 'edge_marginal_tt' or 'edge_tt'
        :return sp_dict: (dict) shortest path dict. e.g., {'1_2':[1,2],...}
        """
        route_cost = []
        edge_marginal_tt = net_df.edge_marginal_tt.values
        for r_emb in assign_info['link_embedding']:
            r_cost = np.dot(r_emb, edge_marginal_tt)
            route_cost.append(r_cost)

        sp_cost = min(route_cost)
        sp_index = route_cost.index(sp_cost)
        sp_link_embedding = assign_info['link_embedding'][sp_index]

        optimized_assign_demand = []

        sp_assigned_demand = assign_info['assign_demand'][sp_index]

        edge_flow_shift = np.zeros(len(sp_link_embedding))  # save edge flow shift for network loading
        # update for each route
        for r_i, (route, r_i_demand, r_i_link_embedding) in enumerate(zip(assign_info['route_list'],
                                                                          assign_info['assign_demand'],
                                                                          assign_info['link_embedding'])):
            r_i_cost = route_cost[r_i]
            if r_i_demand > 0 and r_i != sp_index and r_i_cost > sp_cost:

                #assert round(r_i_cost, 6) >= round(sp_cost, 6)  # make sure that non-basic path has bigger cost

                # the links which are used by either π or ˆ π, but not both.
                derivative_links = np.where(sp_link_embedding != r_i_link_embedding, 1, 0)
                derivative_sum = np.dot(derivative_links, net_df['edge_MTT_derivative'])
                theory_shift_demand = (r_i_cost - sp_cost) / derivative_sum  # use round(,6) to make sure sometimes r_i_cost=6.0000321, sp_cost=6.000032
                shift_demand = min(theory_shift_demand, r_i_demand)

                # adjust demand to sp
                sp_assigned_demand += shift_demand
                r_i_demand -= shift_demand

                edge_flow_shift += shift_demand * (sp_link_embedding - r_i_link_embedding)


            # optimized_route_list.append(route)
            optimized_assign_demand.append(r_i_demand)
            # optimized_link_embedding.append(r_i_link_embedding)

        optimized_assign_demand[sp_index] = sp_assigned_demand  # optimized sp assigned demand

        optimized_assign_demand = np.array(optimized_assign_demand)
        total_ass_demand = np.sum(optimized_assign_demand)
        if total_ass_demand == 0:
            raise ValueError("The sum of the array elements is zero, cannot divide by zero.")
        optimized_assign_ratio = optimized_assign_demand / total_ass_demand

        # optimized assign_info
        assign_info['assign_ratio'] = optimized_assign_ratio
        assign_info['assign_demand'] = optimized_assign_demand

        return assign_info, edge_flow_shift

    def path_based_assignment(self, net_df, demand_df, n_iterations):
        """
        static traffic assignment using gradient projection(GP, path-based method) for n iterations
        :param net_df: (pd.DataFrame) netwrok data
        :param demand_df: (pd.DataFrame) od-demand data
        :param n_iterations: number of iterations
        :return sp_dict: (dict) shortest path dict. e.g., {'1_2':[1,2],...}
        """
        global_gap_list = []
        total_tt_list = []

        all_assign_info = {}
        ###### 1. initialize with all-or-nothing assignment ######
        all_od_pairs = list(demand_df.od.values)
        sp_dict, sp_cost_dict = self.get_odpairs_shortest_path(net_df=net_df, od_pairs=all_od_pairs,
                                                               weight='free_flow_time')  # {'1_2':[1,2],.}
        demand_df_dict = dict(zip(demand_df['od'], demand_df['demand']))

        # assign all demand to shortest path
        net_df['edge_flow'] = 0
        for od in all_od_pairs:  # od = '1_2'
            assign_info = {'route_list': [], 'link_embedding': []}

            sp = sp_dict[od]
            demand = demand_df_dict[od]
            sp_edges = self.edges_from_path(s_path=sp)  # ['1_3', '3_4', '4_5', '5_6', '6_2']
            sp_link_embedding = net_df.od.isin(sp_edges).values.astype(int)
            # update edge flow
            net_df['edge_flow'] += demand * sp_link_embedding

            # save all assigned info
            assign_info['route_list'].append(sp)
            assign_info['assign_ratio'] = np.array([1])
            assign_info['assign_demand'] = np.array([demand])
            assign_info['link_embedding'].append(sp_link_embedding)
            all_assign_info[od] = assign_info

        # update edge state (edge tt, marginal tt...) based on 'edge_flow' using bpr function
        net_df = self.update_net_df(net_df)
        local_gaps_dict, global_gap = self.get_relative_gap(net_df=net_df, demand_df=demand_df,
                                                            all_assign_info=all_assign_info)

        total_tt = np.dot(net_df.edge_flow.values, net_df.edge_tt.values)
        total_tt_list.append(total_tt)
        global_gap_list.append(global_gap)

        ###### 2. optimize using path-based method ######
        def remove_zero_assignments(ass_data):
            zero_indices = np.where(ass_data['assign_ratio'] == 0)[0]
            for key in ass_data.keys():
                if isinstance(ass_data[key], list):
                    ass_data[key] = [value for i, value in enumerate(ass_data[key]) if i not in zero_indices]
                elif isinstance(ass_data[key], np.ndarray):
                    ass_data[key] = np.delete(ass_data[key], zero_indices)
            return ass_data

        start_t = time.time()
        for iter in range(n_iterations):
            print('\n*********************************************************************')
            print('************************** iteration = {}****************************'.format(iter))

            ###### Find shortest paths & Adjust route choices toward equilibrium

            for od in all_od_pairs:  # od = '1_2'
                assign_info = all_assign_info[od]
                demand = demand_df_dict[od]
                opt_assign_info, edge_flow_shift = self.optimize_one_od_using_GP(assign_info, net_df, od,
                                                                                 weight='edge_marginal_tt')
                opt_assign_info_remove_zero = remove_zero_assignments(opt_assign_info)
                all_assign_info[od] = opt_assign_info_remove_zero

                # update edge flow and travel time after optimize each od-pair's assignment
                net_df = self.network_loading_using_ass_info(net_df=net_df, edge_flow_shift=edge_flow_shift)

            local_gaps_dict, global_gap = self.get_relative_gap(net_df=net_df, demand_df=demand_df,
                                                                all_assign_info=all_assign_info)
            # print(time.time() - opt_end_t, 'seconds to get gaps')

            global_gap_list.append(global_gap)
            total_tt = np.dot(net_df.edge_flow.values, net_df.edge_tt.values)
            total_tt_list.append(round(total_tt, 2))
            print('global_gap=', global_gap)
            print('global_gap_list=', global_gap_list)
            print('total_tt_list=', total_tt_list)
            iter_t = time.time()
            print('cost', iter_t - start_t, 'seconds for this iteration')
            start_t = iter_t

        return all_assign_info, global_gap_list

    def path_based_assignment_with_fixed_routes(self, net_df, demand_df, n_iterations):
        """
        static traffic assignment using gradient projection(GP, path-based method) for n iterations,
        however use a fixed route set in this funcion

        :param net_df: (pd.DataFrame) netwrok data
        :param demand_df: (pd.DataFrame) od-demand data
        :param n_iterations: number of iterations
        :return sp_dict: (dict) shortest path dict. e.g., {'1_2':[1,2],...}
        """
        global_gap_list = []
        total_tt_list = []

        # self.route_info

        all_assign_info = {}
        ###### 1. initialize with all-or-nothing assignment ######
        all_od_pairs = list(demand_df.od.values)
        demand_df_dict = dict(zip(demand_df['od'], demand_df['demand']))

        # assign all demand to shortest path
        net_df['edge_flow'] = 0
        for od in all_od_pairs:  # od = '1_2'
            assign_info = {}
            # {'route_list':[[1,2],...], 'assign_ratio': np.array([0.5, ...]),
            # 'assign_demand': np.array([500, ...]),'link_embedding': [array([0, 0, 1, 0,...]),...]}

            demand = demand_df_dict[od]
            assign_info['route_list'] = self.route_info[od]['route_list']
            assign_info['link_embedding'] = self.route_info[od]['link_embedding']

            # update edge flow
            net_df['edge_flow'] += demand * assign_info['link_embedding'][0]

            # save all assigned info
            assign_ratio = np.zeros(len(assign_info['route_list']))
            assign_ratio[0] = 1.
            assign_demand = np.zeros(len(assign_info['route_list']))
            assign_demand[0] = demand
            assign_info['assign_ratio'] = assign_ratio
            assign_info['assign_demand'] = assign_demand

            all_assign_info[od] = assign_info

        # update edge state (edge tt, marginal tt...) based on 'edge_flow' using bpr function
        net_df = self.update_net_df(net_df)
        local_gaps_dict, global_gap = self.get_relative_gap(net_df=net_df, demand_df=demand_df,
                                                            all_assign_info=all_assign_info)

        total_tt = np.dot(net_df.edge_flow.values, net_df.edge_tt.values)
        total_tt_list.append(total_tt)
        global_gap_list.append(global_gap)

        ###### 2. optimize using path-based method ######

        start_t = time.time()
        for iter in range(n_iterations):
            print('\n*********************************************************************')
            print('************************** iteration = {}****************************'.format(iter))

            ###### Find shortest paths & Adjust route choices toward equilibrium
            for od in all_od_pairs:  # od = '1_2'
                assign_info = all_assign_info[od]
                demand = demand_df_dict[od]
                opt_assign_info, edge_flow_shift = self.optimize_one_od_using_GP_with_fixed_routes(assign_info,
                                                                                                   net_df,
                                                                                                   od,
                                                                                                   weight='edge_marginal_tt')

                all_assign_info[od] = opt_assign_info

                # update edge flow and travel time after optimize each od-pair's assignment
                net_df = self.network_loading_using_ass_info(net_df=net_df, edge_flow_shift=edge_flow_shift)

            local_gaps_dict, global_gap = self.get_relative_gap(net_df=net_df, demand_df=demand_df,
                                                                all_assign_info=all_assign_info)
            # print(time.time() - opt_end_t, 'seconds to get gaps')

            global_gap_list.append(global_gap)
            total_tt = np.dot(net_df.edge_flow.values, net_df.edge_tt.values)
            total_tt_list.append(round(total_tt, 2))
            print('global_gap=', global_gap)
            print('global_gap_list=', global_gap_list)
            print('total_tt_list=', total_tt_list)
            iter_t = time.time()
            print('cost', iter_t - start_t, 'seconds for this iteration')
            start_t = iter_t

        return all_assign_info, global_gap_list

    def network_loading_using_ass_info(self, net_df, edge_flow_shift):
        # update edge flow
        net_df['edge_flow'] += edge_flow_shift

        # update edge state (edge tt, marginal tt...) based on 'edge_flow' using bpr function
        net_df = self.update_net_df(net_df)

        return net_df

    def get_relative_gap(self, net_df, demand_df, all_assign_info):
        """
        get global gap and relative gap for each od

        :param all_assign_infoo: (dict) all assignment solution. e.g., {
                                '1_2':{'route_list':[[1,2],...], 'assign_ratio': np.array([0.5, ...]),
                                       'assign_demand': np.array([500,...]), 'link_embedding': [array([0, 1,...]),...]},
                                '2_10':{,,,},...}
        :param net_df: (pd.DataFrame) netwrok data
        :param od_pair: (list) one od pair list. e.g., '1_2'

        :return local_gaps_dic: (dict)
        :return global_gap:
        """
        ##### get relative gap for each od/agent
        all_od_pairs = list(demand_df.od.values)
        demand_df_dict = dict(zip(demand_df['od'], demand_df['demand']))
        sp_dict, sp_cost_dict = self.get_odpairs_shortest_path(net_df=net_df, od_pairs=all_od_pairs,
                                                               weight='edge_marginal_tt')  # {'1_2':[1,2],.}

        local_gaps_dict = {}  # {'1_2':0.2,...}
        SP_COST = 0  # shortest path cost
        TS_COST = np.dot(net_df['edge_marginal_tt'], net_df['edge_flow'])  # current total assignment cost

        edge_marginal_tt = net_df['edge_marginal_tt'].values
        # for agent_id, action in enumerate(actions):
        for od in all_od_pairs:
            ass_info = all_assign_info[od]
            demand = demand_df_dict[od]
            # {'route_list':[[1,2],...], 'assign_ratio': np.array([0.5, ...]),
            # 'assign_demand': np.array([500, ...]),'link_embedding': [array([0, 0, 1, 0,...]),...]}

            sp_c = sp_cost_dict[od] * demand  # shortest path cost
            SP_COST += sp_cost_dict[od] * demand

            #### get current assignment cost of each agent/od
            current_cost = np.sum([np.dot(ass_info['assign_demand'][k] * link_emb, edge_marginal_tt)
                                   for k, link_emb in enumerate(ass_info['link_embedding'])])

            local_gap = (current_cost / sp_c) - 1 
            # print('ass_demand:', ass_demand, 'demand:', demand)
            # print('current_cost=',current_cost, 'sp_c:', sp_c, 'local_gap:', local_gap)
            local_gaps_dict[od] = local_gap

        global_gap = (TS_COST / SP_COST) - 1

        # print('TS_COST=', TS_COST, ' SP_COST=', SP_COST)
        # print('local_gaps_dict:', local_gaps_dict)
        # print('global_gap:', global_gap)

        return local_gaps_dict, global_gap


class Config():
    def __init__(self, net_df, demand_df, all_od_paths, action_space, batch=None):
        self.net_df = net_df
        self.demand_df = demand_df
        self.all_od_paths = all_od_paths
        self.action_space = action_space
        self.batch = batch


def get_path_based_solution():
    from load_net_multi_agent_config import load_SiouxFalls_so_action_set_multi_agent_config
    args_abs_path = os.path.join(os.getcwd(), 'SiouxFallNetData')
    env_args = load_SiouxFalls_so_action_set_multi_agent_config(file_path=args_abs_path)
    env = STA_Continuous_MultiAgent_Env_V1(args=env_args)
    print(env.action_space)

    print(len(list(env.demand_df.od.values)))
    print('list(env.demand_df.od.values):', list(env.demand_df.od.values))

    env.reset()
    # full demand
    all_ass_info, global_gap_l = env.path_based_assignment(net_df=env.net_df, demand_df=env.full_demand_df,
                                                           n_iterations=100)
    # save so data
    env.full_demand_df.to_csv('SiouxFallNetData/SO_Data/full_demand_df.csv')
    with open('SiouxFallNetData/SO_Data/ass_info_full_demand.pkl', 'wb') as f:
        pickle.dump((all_ass_info, global_gap_l), f)

    s_t = time.time()
    for i in range(1000):
        print('\n==================================')
        print('======== num of demand pair =', i, ' ========')
        seed = i
        env.reset(seed=i)
        env.demand_df.to_csv(f'SiouxFallNetData/SO_Data/demand_df_{i}.csv')
        all_ass_info, global_gap_l = env.path_based_assignment(net_df=env.net_df, demand_df=env.demand_df,
                                                               n_iterations=100)
        with open(f'SiouxFallNetData/SO_Data/ass_info_{i}.pkl', 'wb') as f:
            pickle.dump((all_ass_info, global_gap_l), f)
        end_t = time.time()
        print(end_t - s_t, ' seconds num of demand pair =', i, '!!!!!!!!!!!!!')
        s_t = end_t



def run_path_based_method_and_save_result():
    from load_net_multi_agent_config import load_SiouxFalls_so_action_set_multi_agent_config
    args_abs_path = os.path.join(os.getcwd(), 'SiouxFallNetData')
    env_args = load_SiouxFalls_so_action_set_multi_agent_config(file_path=args_abs_path)
    env = STA_Continuous_MultiAgent_Env_V1(args=env_args)
    print(env.action_space)

    print(len(list(env.demand_df.od.values)))
    print('list(env.demand_df.od.values):', list(env.demand_df.od.values))

    env.reset()
    # full demand
    all_ass_info, global_gap_l = env.path_based_assignment(net_df=env.net_df, demand_df=env.full_demand_df,
                                                           n_iterations=100)
    # save so data
    env.full_demand_df.to_csv('SiouxFallNetData/SO_Data/full_demand_df.csv')
    with open('SiouxFallNetData/SO_Data/ass_info_full_demand.pkl', 'wb') as f:
        pickle.dump((all_ass_info, global_gap_l), f)

    s_t = time.time()
    for i in range(1000):
        print('\n==================================')
        print('======== num of demand pair =', i, ' ========')
        seed = i
        env.reset(seed=i)
        env.demand_df.to_csv(f'SiouxFallNetData/SO_Data/demand_df_{i}.csv')
        all_ass_info, global_gap_l = env.path_based_assignment(net_df=env.net_df, demand_df=env.demand_df,
                                                               n_iterations=100)
        with open(f'SiouxFallNetData/SO_Data/ass_info_{i}.pkl', 'wb') as f:
            pickle.dump((all_ass_info, global_gap_l), f)
        end_t = time.time()
        print(end_t - s_t, ' seconds num of demand pair =', i, '!!!!!!!!!!!!!')
        s_t = end_t

    marginal_tt_dict = {}
    test_od_pairs = ['2_1', '17_4', '3_8']
    for agent_id in list(env.agent_ids.keys()):  # 0 --> 528
        od = env.agent_ids[agent_id]  # '1_2'
        if od in test_od_pairs:
            r_info = env.route_info[
                od]  # '2_3' --> {'id_list': [0, 1],  'route_list': [[2,3],...], 'link_embedding':...}

            ########## get dynamic obs/features
            marginal_tt = np.array(
                [np.dot(arr, env.net_df['edge_marginal_tt']) for arr in r_info['link_embedding']],
                dtype=np.float32)
            print('od=', od, ' route list=', r_info['route_list'])
            # self.last_marg_tt_dict[od] = marginal_tt  # update last marginal travel time
            tt = np.array(
                [np.dot(arr / r_info['n_links'][i], env.net_df['edge_tt']) for i, arr in
                 enumerate(r_info['link_embedding'])],
                dtype=np.float32)

            marginal_tt_dict[od] = marginal_tt

    sp_dict, sp_length_dict = env.get_odpairs_shortest_path(net_df=env.net_df, od_pairs=test_od_pairs,
                                                            weight='edge_marginal_tt')

    print('marginal_tt_dict:', marginal_tt_dict)
    print('sp_dict:', sp_dict)
    print('sp_length_dict:', sp_length_dict)


    print('end!')



if __name__ == '__main__':
    from load_net_multi_agent_config import load_SiouxFalls_so_action_set_multi_agent_config

    args_abs_path = os.path.join(os.getcwd(), 'SiouxFallNetData')
    env_args = load_SiouxFalls_so_action_set_multi_agent_config(file_path=args_abs_path)
    env = STA_Continuous_MultiAgent_Env_V1(args=env_args)
    print(env.action_space)

    print(len(list(env.demand_df.od.values)))
    print('list(env.demand_df.od.values):', list(env.demand_df.od.values))

    env.reset()
    # full demand
    all_ass_info, global_gap_l = env.path_based_assignment_with_fixed_routes(net_df=env.net_df,
                                                                             demand_df=env.full_demand_df,
                                                                             n_iterations=100)





