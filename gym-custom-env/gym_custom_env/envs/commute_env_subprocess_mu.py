from time import sleep
from collections import namedtuple
import random
import os
import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .Bottleneck_env import Bottleneck_simulation
from .MFD_env import MFD_simulation

user_params = {'lambda': 3, 'gamma': 2,'hetero':1.6}
scenario = 'Trinity' # simulate:'NT': no toll, 'CP': congestion price, 'Trinity'
Tstep = 1 # discretization in one day 
deltaP = 0.05
# deltaP = 0.2

RBTD = 100
Plot = False
verbose = True
# Policy: False, uniform, personalizedCap, personalizedNocap
# numOfusers = 8625

marketPrice = 1 # initial market price
gamma = 0.02
state_aggravate = 5 #specify the interval for state aggregation
# only applies if scenario is CP
if scenario == 'CP':
     allowance = {'policy': 'personalization','ctrl':1.125,'cap':float("inf")}
else:
     allowance = {'policy': False,'ctrl':1.048,'cap':float("inf")}
marketPrice = 1

CV = False
if scenario == 'NT':
     CV = False
#1.02557013 327.8359478  371.82177488
unusual = {"unusual":False, 'day':10,
            "read":'Trinitytt.npy', 'ctrlprice':2.63,
            'regulatesTime':415, 'regulateeTime':557,
            "dropstime":360, "dropetime":480,
            "FTC": 0.05, "AR":0.0}

storeTT = {'flag':False,"ttfilename":'Trinitylumptt'}

class SubProcess_CommuteEnv_mu(gym.Env): # env reset for each training/testing
        # Define initial parameters for the environment
    def __init__(self, 
                 simulation_day_num = 50, 
                 save_episode_freq = 30,  # check the save episode frequency for evaluation
                 save_dir = "./train_result/", 
                 state_shape = (5, int(12*60/5)),
                 action_shape = (3, ), 
                 Supply_model = "MFD",
                 Initialization = "random", 
                 Reward_scheme = "Weighted", # Add_constant, Weighted, fftt, 
                 Random_mode = False, 
                 Absolute_change_mode = False, 
                 Toll_type = "normal",
                 episode_in_one_eval = 5, 
                 Mode = "simulation", # simulation, eval, train
                 _choiceInterval  = 60, 
                 reward_weight = 1.0, 
                #  seed_value =  random.randint(0, 99999), 
                 Allocation =  {'AR':0.00269,'way':'continuous','FTCs': 0.05,'FTCb':0.05,'PTCs': 0.00,'PTCb':0.00,"Decaying": False},
                 input_save_dir = "tmp/",
                 action_weights  = (1,), 
                 std_weights = 1,
                 pt_weights = 10,
                 numOfusers = 7000
                 ):
        
        super().__init__()
        self.params = {'alpha':1.1, 'omega':0.9, 'theta':5*10**(-1), 'tao':90, 'Number_of_user':3700 } # alpha is unused
        self.supply_model = Supply_model
        self._choiceInterval = _choiceInterval
        self.simulation_day_num = simulation_day_num
        self.save_episode_freq = save_episode_freq
        self.toll_type = Toll_type
        self.initialization = Initialization
        self.reward_scheme = Reward_scheme
        self.absolute_change_mode = Absolute_change_mode
        self.mode = Mode # if mode = "simulatiom", the action is fixed given by main.py
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.action_space =  spaces.Box(low = -np.ones(action_shape[0]), 
                                        high=np.ones(action_shape[0]), 
                                        shape=action_shape, dtype=np.float32)
        # self.observation_space = spaces.Box(low = -99999, high = 99999,\
        #                                     shape = state_shape, dtype = np.float64) # include more observations for a broader observation space
        # self.observation_space = gym.spaces.Dict({
        #     "tt": gym.spaces.Box(low = -9999, high = 9999, shape = (state_shape[1], ), dtype = np.float64),
        #     "accumulation": gym.spaces.Box(low = -9999, high = 9999, shape = (state_shape[1], ), dtype = np.float64),
        #     "buy": gym.spaces.Box(low = -9999, high = 9999, shape = (state_shape[1], ), dtype = np.float64),
        #     "sell": gym.spaces.Box(low = -9999, high = 9999, shape = (state_shape[1], ), dtype = np.float64),
        #     "current_A": gym.spaces.Box(low = 0, high = 7, shape = (1, ), dtype = np.float64),
        #     "sigma": gym.spaces.Box(low = 0, high = 10, shape = (1,), dtype = np.float64),
        # })

        self.observation_space = gym.spaces.Dict({
            "accumulation": gym.spaces.Box(low = -9999, high = 9999, shape = (state_shape[1], ), dtype = np.float64),
            "current_mu": gym.spaces.Box(low = 300, high = 540, shape = (1, ), dtype = np.float64),
            "price": gym.spaces.Box(low = 0, high = 4, shape = (1,), dtype = np.float64),
        })
        self.allocation = Allocation
        self.render_mode = False
        self.episode_in_one_eval = episode_in_one_eval
        self.day = 0
        self.episode = 0 
        self.eval_episode = 0
        self.reward_weight = reward_weight
        self.input_save_dir = input_save_dir
        self.action_weights = action_weights
        self.std_weights = std_weights
        self.pt_weights = pt_weights
        self.numOfusers = numOfusers

        self.random = Random_mode # random policy
        self.tt_one_eps = [] # daily average travel time
        self.sw_one_eps = [] # social welfare
        self.mp_one_eps = [] # market price
        self.rw_one_eps = [] # step reward
        self.action_one_eps = [] # action change
        self.toll_one_eps = [] # toll profile
        self.pt_one_eps = [] # pt profile
        self.tt_interval_one_eps = np.zeros((self.simulation_day_num+1, 144))

        self.tt_all_eps = [] # in all episodes
        self.tt_last_5_day_all_eps = [] # in all episodes
        self.sw_all_eps = [] # in all episodes
        self.mp_all_eps = [] 
        self.rw_all_eps = []
        self.action_all_eps = []
        self.toll_all_eps = []
        self.flow_all_eps = []
        self.usertrade_all_eps = []
        self.tokentrade_all_eps = []
        self.convergence_all_eps = []
        self.pt_all_eps = [] # toll profile
        self.tt_interval_all_eps = []

        self.tt_interval_one_eval = [] 
        self.tt_one_eval = []   # record tt in 5-episode evaluation
        self.tt_last_5_day_one_eval = []  
        self.sw_one_eval = []  
        self.mp_one_eval = []  
        self.rw_one_eval = []  
        self.action_one_eval = []  
        self.toll_one_eval = []  
        self.pt_one_eval = []  
        self.flow_one_eval = []
        
        self.tt_interval_all_eps_eval = []
        self.tt_all_eps_eval = []  
        self.tt_last_5_day_all_eps_eval = [] # last-5-day AITT in all eval episodes
        self.sw_all_eps_eval = [] 
        self.mp_all_eps_eval = [] 
        self.rw_all_eps_eval = []
        self.action_all_eps_eval = []
        self.toll_all_eps_eval = []
        self.flow_all_eps_eval = []
        self.usertrade_all_eps_eval = []
        self.tokentrade_all_eps_eval = []
        self.convergence_all_eps_eval = []
        self.pt_all_eps_eval = [] # toll profile

        self.save_dir = save_dir
        self.num_envs = 1
        # self.seed = seed_value
        self.last_AITT_daily = 35
        self.std_weight = std_weights

    # This method generates a new starting state often with some randomness 
    # to ensure that the agent explores the state space and learns a generalised policy 
    # about the environment.   
    def reset(self, seed = None, options =None): # env reset for each episode
        # print(" reset ")
        # print(" self.mode ", self.mode)
        super().reset()
        # self.set_seed(self.seed)
        self.day = 0
        self.tt_one_eps = [] # daily average travel time in one episode
        self.sw_one_eps = [] # social welfare
        self.mp_one_eps = [] # market price
        self.rw_one_eps = [] # step reward
        self.action_one_eps = [] # action change
        self.toll_one_eps = [] # A profile
        self.pt_one_eps = [] # pt change
        self.tt_interval_one_eps = np.zeros((self.simulation_day_num+1, 144))

        self.toll_A = 3.62677671
        self.toll_mu = 120*(random.random()*2-1) + 420
        self.toll_sigma = 63.50159324
        
        if self.supply_model == "MFD":  
            self.sim = MFD_simulation(_numOfdays = self.simulation_day_num, 
                                        _user_params = user_params,
                                        _scenario = scenario,
                                        _allowance = allowance, 
                                        _marketPrice = marketPrice, 
                                        _allocation = self.allocation,
                                        _deltaP = deltaP, 
                                        _numOfusers= self.numOfusers, 
                                        _RBTD = RBTD, 
                                        _Tstep=Tstep, 
                                        _Plot = Plot, 
                                        _verbose = verbose, 
                                        _unusual = unusual, 
                                        _storeTT=storeTT, 
                                        _CV=CV, 
                                        save_dfname = self.save_dir + "NT",
                                        toll_type = self.toll_type, 
                                        _choiceInterval = 60,
                                        _input_save_dir = self.input_save_dir, 
                                   )
        elif self.supply_model == "Bottleneck":
            self.sim = Bottleneck_simulation(
                            _numOfdays = self.simulation_day_num, 
                            _user_params = user_params,
                            _scenario = scenario,
                            _allowance = allowance, 
                            _marketPrice = marketPrice, 
                            _allocation = self.allocation,
                            _deltaP = deltaP, 
                            _numOfusers = self.numOfusers, 
                            _RBTD = RBTD,
                            _Tstep = Tstep, 
                            _Plot = Plot, 
                            # _seed = self.seed_value, 
                            _verbose = verbose,
                            _unusual = unusual, 
                            _storeTT = storeTT, _CV=CV, 
                            save_dfname = './output/RL_Bottleneck/Trinity', 
                            toll_type = self.toll_type,
                            _choiceInterval = self._choiceInterval
                        )
            
        self.sim.toll_parameter = np.array([self.toll_A, self.toll_mu, self.toll_sigma]) 
        self.sim.users.toll_parameter = np.array([self.toll_A, self.toll_mu, self.toll_sigma])       
        tt_state, accumulation_state, sell_state, buy_state, pt_share_number, market_price, sw, tt_util, sde_util, sdl_util, ptwaiting_util, I_util, userBuy_util, userSell_util, fuelcost_util = self.sim.RL_simulateOneday(self.day, state_aggravate, self.state_shape) # 5 days social welfare
        self.tt_interval_one_eps[0] = tt_state
        # sigma_tt_interval = np.zeros(144)
        # observation = {
        #    "tt": np.array(tt_state, dtype = np.float64), 
        #    "accumulation": np.array(accumulation_state, dtype = np.float64), 
        #    "buy": np.array(buy_state, dtype = np.float64), 
        #    "sell": np.array(sell_state, dtype = np.float64),
        #    "current_A": np.array([self.toll_A], dtype = np.float64),
        #    "sigma": np.array([sigma_tt_interval],  dtype = np.float64),
        # }
        observation = {
           "accumulation": np.array(accumulation_state, dtype = np.float64), 
           "current_mu": np.array([self.toll_mu], dtype = np.float64),
           "price": np.array([market_price],  dtype = np.float64),
        #    "sigma": np.array(sigma_tt_interval,  dtype = np.float64),
        }

        # print(" reset day ", self.day)
        # print(" self.tt_interval_one_eps[0] ", self.tt_interval_one_eps[0])   
        # print(" sigma_tt_interval ", sigma_tt_interval)
        # print(" ")

        self.tt_one_eps.append(np.mean(self.sim.flow_array[self.day, :, 2]))
        self.sw_one_eps.append(sw)
        self.mp_one_eps.append(market_price)
        self.toll_one_eps.append(self.sim.toll_parameter)
        self.pt_one_eps.append(pt_share_number)
        AITT_daily = np.mean(self.sim.flow_array[self.day, :, 2]) 
        tmp = self.sim.flow_array[self.day]
        tmp_2 = tmp[tmp[:,0] != -1]
        AITT_car_only = np.mean(tmp_2[:, 2]) # calculate the average travel time on day 

        info = {"toll_parameter_A" : self.toll_A,
                "pt_share_number": pt_share_number, 
                "sw": sw, 
                "market_price": market_price, 
                "AITT_daily": AITT_daily,
                "AITT_car_only": AITT_car_only, 
                "tt_util": tt_util, 
                "sde_util": sde_util, 
                "sdl_util": sdl_util, 
                "ptwaiting_util": ptwaiting_util, 
                "I_util": I_util, 
                "userBuy_util": userBuy_util, 
                "userSell_util": userSell_util, 
                "fuelcost_util": fuelcost_util
                }

        # print(" observation " ,observation)
        
        return observation, info
    
    def step(self, action):
        done = False
        timeofday = np.arange(self.sim.hoursInA_Day*60) # the toll fees of the day
        
        if self.toll_mu + self.action_weights[0] * action[0] < 300:
            self.toll_mu = 300
        elif self.toll_mu + self.action_weights[0] * action[0]> 540:
            self.toll_mu = 540
        else:
            self.toll_mu += self.action_weights[0] * action[0]
                        
            # if self.toll_type == "step":
            #     toll_parameter = np.array([self.toll_A, 120*self.toll_mu+420, 5*self.toll_sigma+20,])
            #     self.sim.toll = np.maximum(self.sim.steptoll_fxn(timeofday, *toll_parameter), 0)
            #     self.sim.toll = np.around(self.sim.toll, 2)

            if self.toll_type == "normal":
                self.sim.toll_parameter = np.array([self.toll_A, self.toll_mu, self.toll_sigma])
                self.sim.users.toll_parameter = np.array([self.toll_A, self.toll_mu, self.toll_sigma])
                self.sim.toll = np.repeat(np.maximum(self.sim.bimodal(timeofday[np.arange(0,self.sim.hoursInA_Day*60,self.sim.Tstep)]),0),self.sim.Tstep)
                self.sim.toll = np.around(self.sim.toll, 2)
    
        if (self.mode == "simulation"):
            self.sim.toll_parameter = np.array([self.toll_A, action[0], self.toll_sigma])
            self.sim.users.toll_parameter =np.array([self.toll_A, action[0], self.toll_sigma])
            self.sim.toll = np.repeat(np.maximum(self.sim.bimodal(timeofday[np.arange(0,self.sim.hoursInA_Day*60,self.sim.Tstep)]),0),self.sim.Tstep)
            self.sim.toll = np.around(self.sim.toll, 2)

        tt_state, accumulation_state, sell_state, buy_state, pt_share_number, market_price, sw, tt_util, sde_util, sdl_util, ptwaiting_util, I_util, userBuy_util, userSell_util, fuelcost_util = self.sim.RL_simulateOneday(self.day, state_aggravate, self.state_shape) # 5 days social welfare
        
        self.tt_interval_one_eps[self.day + 1] = tt_state # travel time on that interval

        if self.day < 4:
            sigma_tt_interval = np.std(self.tt_interval_one_eps[:self.day+2], axis=0) # sigm
        else:
            sigma_tt_interval = np.std(self.tt_interval_one_eps[self.day-3:self.day+2], axis=0)
        
        observation  = {
           "accumulation": np.array(accumulation_state, dtype = np.float64), 
           "current_mu": np.array([self.toll_mu], dtype = np.float64),
           "price": np.array([market_price],  dtype = np.float64),
        }
        

        AITT_daily = np.mean(self.sim.flow_array[self.day, :, 2]) # calculate the average travel time on day 
        tmp = self.sim.flow_array[self.day]
        tmp_2 = tmp[tmp[:,0] != -1]
        AITT_car_only = np.mean(tmp_2[:, 2]) # calculate the average travel time on day 

        if self.reward_scheme == "Shifting":
            reward = self.last_AITT_daily - AITT_daily

        elif self.reward_scheme == "Add_constant": 
            if self.supply_model == "Bottleneck":
                reward =  - AITT_daily + 30
            else:
                reward = - AITT_daily + 45

        elif self.reward_scheme == "fftt_minus_pt":
            # reward = - AITT_daily/self.sim.fftt  - self.std_weights *  np.mean(sigma_tt_interval)
            # reward = - AITT_daily/self.sim.fftt
            reward = (- AITT_daily/self.sim.fftt - self.pt_weights *  pt_share_number/self.numOfusers) /100

        elif self.reward_scheme == "fftt_plus_pt":
            reward = (- AITT_daily/self.sim.fftt + self.pt_weights *  pt_share_number/self.numOfusers)/100

        elif self.reward_scheme == "triangle":
            # reward = - AITT_daily/self.sim.fftt  - self.std_weights *  np.mean(sigma_tt_interval)
            # reward = - AITT_daily/self.sim.fftt
            if pt_share_number > self.numOfusers/10:
                pt_reward = - (pt_share_number - self.numOfusers/10)/self.numOfusers
            else:
                pt_reward = - (self.numOfusers/10 - pt_share_number)/self.numOfusers
            
            reward = (-AITT_daily/self.sim.fftt + self.pt_weights * pt_reward)/100

        elif self.reward_scheme == "Weighted_reward":
            reward = (-AITT_daily/self.sim.fftt  - self.std_weights * np.mean(sigma_tt_interval))/self.reward_weight
            # if self.day>24:
            #     reward *= self.reward_weight

        self.last_AITT_daily = AITT_daily

        if self.action_shape[0] == 1:
            self.action_one_eps.append([action[0]])
        elif self.action_shape[0] == 2:
            self.action_one_eps.append([action[0], action[1]])
        else:
            self.action_one_eps.append([action[0], action[1], action[2]])

        self.toll_one_eps.append(self.sim.toll_parameter)
        self.pt_one_eps.append(pt_share_number)
        self.rw_one_eps.append(reward)
        self.sw_one_eps.append(sw)
        self.mp_one_eps.append(market_price)
        self.tt_one_eps.append(AITT_daily)

        info = {
                    "toll_parameter_A" : self.sim.toll_parameter[0],
                    "rw": reward,
                    "pt_share_number":pt_share_number, 
                    "sw": sw, 
                    "market_price": market_price, 
                    "AITT_daily": AITT_daily,
                    "AITT_car_only": AITT_car_only,
                    "tt_util": tt_util, 
                    "sde_util": sde_util, 
                    "sdl_util": sdl_util, 
                    "ptwaiting_util": ptwaiting_util, 
                    "I_util": I_util, 
                    "userBuy_util": userBuy_util, 
                    "userSell_util": userSell_util, 
                    "fuelcost_util": fuelcost_util
                }
        
        if (self.mode == "train"):
            # if it is the last step in the episode
            if self.day == self.simulation_day_num-1:
                self.tt_all_eps.append(np.array(self.tt_one_eps)) # record 30-day simulation AITT
                self.tt_interval_all_eps.append(np.array(self.tt_interval_one_eps)) # record 30-day simulation AITT
                self.tt_last_5_day_all_eps.append(np.mean(np.array(self.tt_one_eps)[-5:])) # only record the average AITT of last 5 days
                self.sw_all_eps.append(np.array(self.sw_one_eps))
                self.mp_all_eps.append(np.array(self.mp_one_eps))
                self.rw_all_eps.append(np.array(self.rw_one_eps))
                self.action_all_eps.append(np.array(self.action_one_eps))
                self.toll_all_eps.append(np.array(self.toll_one_eps))
                # self.flow_all_eps.append(np.array(self.sim.flow_array))
                self.tokentrade_all_eps.append(np.array(self.sim.tokentrade_array))
                self.usertrade_all_eps.append(np.array(self.sim.usertrade_array))
                self.convergence_all_eps.append(np.array(self.sim.users.norm_list))
                self.pt_all_eps.append(np.array(self.pt_one_eps))
                # print(" training episode ", self.episode)

                if  (self.episode+1) % self.save_episode_freq == 0:
                    np.save((self.save_dir+"tt_interval.npy"),(np.array(self.tt_interval_all_eps)))
                    np.save((self.save_dir+"toll.npy"), self.get_toll())
                    np.save((self.save_dir+"tt_last_5_day.npy"), self.get_tt_last_5_day())
                    np.save((self.save_dir+"tt.npy"), self.get_tt())
                    np.save((self.save_dir+"pt.npy"), self.get_pt())
                    np.save((self.save_dir+"sw.npy"), self.get_sw())
                    np.save((self.save_dir+"mp.npy"), self.get_mp())
                    np.save((self.save_dir+"rw.npy"), self.get_rw())
                    np.save((self.save_dir+"action.npy"), self.get_action())    
                    # np.save((self.save_dir+"flow.npy"), self.get_flow())    
                    np.save((self.save_dir+"usertrade.npy"), self.get_usertrade())    
                    np.save((self.save_dir+"tokentrade.npy"), self.get_tokentrade())   
                    np.save((self.save_dir+"convergence.npy"), self.get_convergence())  

                self.episode =  self.episode +1   

                done = True

        elif (self.mode == "simulation"):
            # if it is the last step in the episode
            if self.day == self.simulation_day_num-1:
                self.tt_all_eps.append(np.array(self.tt_one_eps)) # record 30-day simulation AITT
                self.tt_interval_all_eps.append(np.array(self.tt_interval_one_eps)) # record 30-day simulation AITT
                self.tt_last_5_day_all_eps.append(np.mean(np.array(self.tt_one_eps)[-5:])) # only record the average AITT of last 5 days
                self.sw_all_eps.append(np.array(self.sw_one_eps))
                self.mp_all_eps.append(np.array(self.mp_one_eps))
                self.rw_all_eps.append(np.array(self.rw_one_eps))
                self.action_all_eps.append(np.array(self.action_one_eps))
                self.toll_all_eps.append(np.array(self.toll_one_eps))
                self.flow_all_eps.append(np.array(self.sim.flow_array))
                self.tokentrade_all_eps.append(np.array(self.sim.tokentrade_array))
                self.usertrade_all_eps.append(np.array(self.sim.usertrade_array))
                self.convergence_all_eps.append(np.array(self.sim.users.norm_list))
                self.pt_all_eps.append(np.array(self.pt_one_eps))
                # print(" training episode ", self.episode)

                if  (self.episode+1) % self.save_episode_freq == 0:
                    np.save((self.save_dir+"tt_interval.npy"),(np.array(self.tt_interval_all_eps)))
                    np.save((self.save_dir+"toll.npy"), self.get_toll())
                    np.save((self.save_dir+"tt_last_5_day.npy"), self.get_tt_last_5_day())
                    np.save((self.save_dir+"tt.npy"), self.get_tt())
                    np.save((self.save_dir+"pt.npy"), self.get_pt())
                    np.save((self.save_dir+"sw.npy"), self.get_sw())
                    np.save((self.save_dir+"mp.npy"), self.get_mp())
                    np.save((self.save_dir+"rw.npy"), self.get_rw())
                    np.save((self.save_dir+"action.npy"), self.get_action())    
                    np.save((self.save_dir+"flow.npy"), self.get_flow())    
                    np.save((self.save_dir+"usertrade.npy"), self.get_usertrade())    
                    np.save((self.save_dir+"tokentrade.npy"), self.get_tokentrade())   
                    np.save((self.save_dir+"convergence.npy"), self.get_convergence())  

                self.episode =  self.episode +1   

                done = True

        if self.mode == "eval":
            if self.day == self.simulation_day_num-1:
                # print(" eval_episode ", self.eval_episode)
                self.tt_one_eval.append(np.array(self.tt_one_eps))
                self.tt_interval_one_eval.append(np.array(self.tt_interval_one_eps))
                self.tt_last_5_day_one_eval.append(np.mean(np.array(self.tt_one_eps)[-5:])) # only record the average AITT of last 5 days
                self.sw_one_eval.append(np.array(self.sw_one_eps))
                self.mp_one_eval.append(np.array(self.mp_one_eps))
                self.rw_one_eval.append(np.array(self.rw_one_eps))
                self.action_one_eval.append(np.array(self.action_one_eps))
                self.toll_one_eval.append(np.array(self.toll_one_eps))
                self.pt_one_eval.append( np.array(self.pt_one_eps))
                self.flow_one_eval.append(np.array(self.sim.flow_array))

                if (self.eval_episode+1) % self.episode_in_one_eval ==0:
                    # print(" finish eval in one episode :", self.eval_episode)
                    self.tt_all_eps_eval.append(self.tt_one_eval)
                    self.tt_interval_all_eps_eval.append(self.tt_interval_one_eval)
                    self.tt_last_5_day_all_eps_eval.append(np.array(self.tt_last_5_day_one_eval)) # only record the average AITT of last 5 days
                    self.sw_all_eps_eval.append(np.array(self.sw_one_eval))
                    self.mp_all_eps_eval.append(np.array(self.mp_one_eval))
                    self.rw_all_eps_eval.append(np.array(self.rw_one_eval))
                    self.action_all_eps_eval.append(np.array(self.action_one_eval))
                    self.toll_all_eps_eval.append(np.array(self.toll_one_eval))
                    self.pt_all_eps_eval.append(np.array(self.pt_one_eval))
                    self.flow_all_eps_eval.append(np.array(self.flow_one_eval))

                    self.tt_one_eval = []
                    self.tt_interval_one_eval = []
                    self.tt_last_5_day_one_eval = []
                    self.sw_one_eval = []
                    self.mp_one_eval = [] 
                    self.rw_one_eval = []
                    self.action_one_eval = []
                    self.toll_one_eval = []
                    self.pt_one_eval = []
                    self.flow_one_eval = []

                if  (self.eval_episode+1 )% self.save_episode_freq == 0:
                    # print(" save to np ", self.eval_episode)
                    np.save((self.save_dir+"tt_interval.npy"), np.array(self.tt_interval_all_eps_eval))
                    np.save((self.save_dir+"toll.npy"), np.array(self.toll_all_eps_eval))
                    np.save((self.save_dir+"tt_last_5_day.npy"),  np.array(self.tt_last_5_day_all_eps_eval))
                    np.save((self.save_dir+"tt.npy"),  np.array(self.tt_all_eps_eval))
                    np.save((self.save_dir+"pt.npy"),  np.array(self.pt_all_eps_eval))
                    np.save((self.save_dir+"sw.npy"),  np.array(self.sw_all_eps_eval))
                    np.save((self.save_dir+"mp.npy"), np.array(self.mp_all_eps_eval))
                    np.save((self.save_dir+"rw.npy"),  np.array(self.rw_all_eps_eval))
                    np.save((self.save_dir+"action.npy"),  np.array(self.action_all_eps_eval))  
                    np.save((self.save_dir+"flow.npy"),  np.array(self.flow_all_eps_eval))  

                self.eval_episode =  self.eval_episode +1   
                done = True

        self.day += 1
        return observation, reward, done, False, info
    
    def set_seed(self, seed_value): 
        random.seed(seed_value)                  
        np.random.seed(seed_value)
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.backends.cudnn.deterministic = True
        self.seed_value = 333
        self.seed = random.randint(0, 99999)

        # defining functions to get statistics from environment
    def get_day(self):
        return self.day

    def get_tt(self):
        return np.array(self.tt_all_eps)

    def get_sw(self):
        return np.array(self.sw_all_eps)

    def get_mp(self): # get market price
        return np.array(self.mp_all_eps)

    def get_rw(self): 
        return np.array(self.rw_all_eps)

    def get_action(self): 
        return np.array(self.action_all_eps)
    
    def get_toll(self): 
        return np.array(self.toll_all_eps)
    
    def get_flow(self): 
        return np.array(self.flow_all_eps)

    def get_tokentrade(self): 
        return np.array(self.tokentrade_all_eps)
    
    def get_tt_last_5_day(self): 
        return np.array(self.tt_last_5_day_all_eps)
    
    def get_usertrade(self):
        return np.array(self.usertrade_all_eps)

    def get_convergence(self):
        return np.array(self.convergence_all_eps)
    
    def get_pt(self):
        return np.array(self.pt_all_eps)
    
    def render(self):
        pass

    def close(self):
        pass

    def _get_info(self):
        return {}
