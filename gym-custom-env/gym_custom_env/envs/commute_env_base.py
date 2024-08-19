import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import os

class CommuteEnvBase(gym.Env):
    def __init__(self, 
                 simulation_day_num=50, 
                 save_episode_freq=30,  
                 save_dir="./train/", 
                 state_shape=(5, int(12*60/5)),
                 action_shape=(3,), 
                 supply_model="MFD",
                 initialization="random", 
                 reward_scheme="Weighted", 
                 absolute_change_mode=False, 
                 toll_type="normal",
                 episode_in_one_eval=5, 
                 mode="simulation", 
                 choice_interval=60, 
                 reward_weight=1.0, 
                 allocation=None,
                 input_save_dir="tmp/",
                 action_weights=(2,), 
                 std_weights=1,
                 pt_weights=10,
                 num_of_users=7500,
                 capacity=7000,
                 user_params={'lambda': 3, 'gamma': 2, 'hetero': 1.6}
                ):
        
        super().__init__()
        self.simulation_day_num = simulation_day_num
        self.save_episode_freq = save_episode_freq
        self.save_dir = save_dir
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.supply_model = supply_model
        self.initialization = initialization
        self.reward_scheme = reward_scheme
        self.absolute_change_mode = absolute_change_mode
        self.toll_type = toll_type
        self.episode_in_one_eval = episode_in_one_eval
        self.mode = mode
        self.choice_interval = choice_interval
        self.reward_weight = reward_weight
        self.allocation = allocation or {'AR': 0.00269, 'way': 'continuous', 'FTCs': 0.05, 'FTCb': 0.05, 'PTCs': 0.00, 'PTCb': 0.00, "Decaying": False}
        self.input_save_dir = input_save_dir
        self.action_weights = action_weights
        self.std_weights = std_weights
        self.pt_weights = pt_weights
        self.num_of_users = num_of_users
        self.capacity = capacity
        self.user_params = user_params

        self.day = 0
        self.episode = 0
        self.last_AITT_daily = 35

        self.reset_episode_variables()
        self.initialize_data_storage()
        self.define_spaces()

    def define_spaces(self):
        """Abstract method to define observation and action spaces."""
        self.action_space = spaces.Box(low=-np.ones(self.action_shape[0]), 
                                        high=np.ones(self.action_shape[0]), 
                                        shape=self.action_shape, 
                                        dtype=np.float32)
        self.observation_space = self._define_observation_space()

    def reset_episode_variables(self):
        """Reset variables related to the current episode."""
        self.tt_one_eps = []
        self.sw_one_eps = []
        self.mp_one_eps = []
        self.rw_one_eps = []
        self.action_one_eps = []
        self.toll_one_eps = []
        self.pt_one_eps = []
        self.tt_interval_one_eps = np.zeros((self.simulation_day_num+1, 144))

    def initialize_data_storage(self):
        """Initialize storage for data across all episodes."""
        self.tt_all_eps = []
        self.sw_all_eps = []
        self.mp_all_eps = []
        self.rw_all_eps = []
        self.action_all_eps = []
        self.toll_all_eps = []
        self.pt_all_eps = []
        self.tt_interval_all_eps = []
        self.tt_last_5_day_all_eps = []

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset()
        self.day = 0
        self.reset_episode_variables()
        self.sim = self._initialize_simulation()
        observation, reward, done, info = self._simulate_day_and_get_observation_info()
        return observation, info

    def step(self, action):
        """Apply action and simulate one step."""
        self._apply_action(action)
        observation, reward, done, info = self._simulate_day_and_get_observation_info()
        if self._is_last_day_of_episode():
            self._handle_end_of_episode()
            self._save_if_needed()
            self.episode += 1
            done = True
        self.day += 1
        return observation, reward, done, False, info

    def _apply_action(self, action):
        raise NotImplementedError

    def _initialize_simulation(self):
        raise NotImplementedError

    def _simulate_day_and_get_observation_info(self):
        raise NotImplementedError

    def _define_observation_space(self):
        raise NotImplementedError

    def _is_last_day_of_episode(self):
        return self.day == self.simulation_day_num - 1

    def _calculate_reward(self, AITT_daily, pt_share_number):
        if self.reward_scheme == "Shifting":
            return self.last_AITT_daily - AITT_daily
        elif self.reward_scheme == "Add_constant":
            return -AITT_daily + 45
        elif self.reward_scheme == "fftt_minus_pt":
            return (-AITT_daily / self.sim.fftt - self.pt_weights * pt_share_number / self.num_of_users) / 100
        elif self.reward_scheme == "fftt_plus_pt":
            return (-AITT_daily / self.sim.fftt + self.pt_weights * pt_share_number / self.num_of_users) / 100
        elif self.reward_scheme == "triangle":
            pt_reward = -abs(pt_share_number - self.num_of_users / 10) / self.num_of_users
            return (-AITT_daily / self.sim.fftt + self.pt_weights * pt_reward) / 100
        elif self.reward_scheme == "Weighted_reward":
            return (-AITT_daily / self.sim.fftt - self.std_weights * np.mean(np.std(self.tt_interval_one_eps[:self.day + 2], axis=0))) / self.reward_weight
        return 0

    def _record_step_data(self, AITT_daily, sw, market_price, pt_share_number, reward):
        self.tt_one_eps.append(AITT_daily)
        self.sw_one_eps.append(sw)
        self.mp_one_eps.append(market_price)
        self.rw_one_eps.append(reward)
        self.action_one_eps.append([self.toll_A, self.toll_mu, self.toll_sigma])
        self.toll_one_eps.append(self.sim.toll_parameter)
        self.pt_one_eps.append(pt_share_number)

    def _handle_end_of_episode(self):
        self.tt_all_eps.append(np.array(self.tt_one_eps))
        self.tt_interval_all_eps.append(np.array(self.tt_interval_one_eps))
        self.tt_last_5_day_all_eps.append(np.mean(np.array(self.tt_one_eps)[-5:]))
        self.sw_all_eps.append(np.array(self.sw_one_eps))
        self.mp_all_eps.append(np.array(self.mp_one_eps))
        self.rw_all_eps.append(np.array(self.rw_one_eps))
        self.action_all_eps.append(np.array(self.action_one_eps))
        self.toll_all_eps.append(np.array(self.toll_one_eps))
        self.pt_all_eps.append(np.array(self.pt_one_eps))

    def _save_if_needed(self):
        if (self.episode + 1) % self.save_episode_freq == 0:
            np.save(os.path.join(self.save_dir, "tt_interval.npy"), np.array(self.tt_interval_all_eps))
            np.save(os.path.join(self.save_dir, "toll.npy"), self.get_toll())
            np.save(os.path.join(self.save_dir, "tt_last_5_day.npy"), self.get_tt_last_5_day())
            np.save(os.path.join(self.save_dir, "tt.npy"), self.get_tt())
            np.save(os.path.join(self.save_dir, "pt.npy"), self.get_pt())
            np.save(os.path.join(self.save_dir, "sw.npy"), self.get_sw())
            np.save(os.path.join(self.save_dir, "mp.npy"), self.get_mp())
            np.save(os.path.join(self.save_dir, "rw.npy"), self.get_rw())
            np.save(os.path.join(self.save_dir, "action.npy"), self.get_action())

    def get_toll(self):
        return np.array(self.toll_all_eps)

    def get_tt_last_5_day(self):
        return np.array(self.tt_last_5_day_all_eps)

    def get_tt(self):
        return np.array(self.tt_all_eps)

    def get_pt(self):
        return np.array(self.pt_all_eps)

    def get_sw(self):
        return np.array(self.sw_all_eps)

    def get_mp(self):
        return np.array(self.mp_all_eps)

    def get_rw(self):
        return np.array(self.rw_all_eps)

    def get_action(self):
        return np.array(self.action_all_eps)
