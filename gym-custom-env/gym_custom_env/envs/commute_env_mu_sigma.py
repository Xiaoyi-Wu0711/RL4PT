import numpy as np
from .commute_env_base import CommuteEnvBase
from .Bottleneck_env import Bottleneck_simulation
from .MFD_env import MFD_simulation
import random
import gymnasium as gym

class SubProcess_CommuteEnv_mu_sigma(CommuteEnvBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.toll_A = 3.65  # Fixed A value
        self.toll_mu = 120 * (random.random() * 2 - 1) + 420
        self.toll_sigma = 10 * (random.random() * 2 - 1) + 60

    def _define_observation_space(self):
        return gym.spaces.Dict({
            "accumulation": gym.spaces.Box(low=-9999, high=9999, shape=(self.state_shape[1],), dtype=np.float64),
            "current_mu": gym.spaces.Box(low=300, high=540, shape=(1,), dtype=np.float64),
            "current_sigma": gym.spaces.Box(low=50, high=70, shape=(1,), dtype=np.float64),
            "price": gym.spaces.Box(low=0, high=4, shape=(1,), dtype=np.float64),
        })

    def _apply_action(self, action):
        # Apply action to modify toll_mu and toll_sigma
        self.toll_mu = np.clip(self.toll_mu + self.action_weights[0] * action[0], 300, 540)
        self.toll_sigma = np.clip(self.toll_sigma + self.action_weights[1] * action[1], 50, 70)

        self.sim.toll_parameter = np.array([self.toll_A, self.toll_mu, self.toll_sigma])
        self.sim.users.toll_parameter = np.array([self.toll_A, self.toll_mu, self.toll_sigma])

        timeofday = np.arange(self.sim.hoursInA_Day * 60)
        self.sim.toll = np.repeat(
            np.maximum(self.sim.bimodal(timeofday[np.arange(0, self.sim.hoursInA_Day * 60, self.sim.Tstep)]), 0),
            self.sim.Tstep)
        self.sim.toll = np.around(self.sim.toll, 2)

    def _initialize_simulation(self):
        self.toll_A = 3.65  # Fixed A value
        self.toll_mu = 120 * (random.random() * 2 - 1) + 420
        self.toll_sigma = 10 * (random.random() * 2 - 1) + 60

        if self.supply_model == "MFD":
            sim = MFD_simulation(
                _numOfdays=self.simulation_day_num,
                _user_params=self.user_params,
                _scenario="Trinity",  # assuming 'Trinity' as a default
                _allowance={"policy": False, "ctrl": 1.048, "cap": float("inf")},
                _marketPrice=1,
                _allocation=self.allocation,
                _deltaP=0.05,
                _numOfusers=self.num_of_users,
                _RBTD=100,
                _Tstep=1,
                _Plot=False,
                _verbose=True,
                _unusual={"unusual": False, 'day': 10, "read": 'Trinitytt.npy', 'ctrlprice': 2.63,
                          'regulatesTime': 415, 'regulateeTime': 557, "dropstime": 360, "dropetime": 480,
                          "FTC": 0.05, "AR": 0.0},
                _storeTT={"flag": False, "ttfilename": 'Trinitylumptt'},
                _CV=False,
                save_dfname=self.save_dir + "NT",
                toll_type=self.toll_type,
                _choiceInterval=self.choice_interval,
                _input_save_dir=self.input_save_dir,
            )
        elif self.supply_model == "Bottleneck":
            sim = Bottleneck_simulation(
                _numOfdays=self.simulation_day_num,
                _user_params=self.params,
                _scenario="Trinity",
                _allowance={"policy": False, "ctrl": 1.048, "cap": float("inf")},
                _marketPrice=1,
                _allocation=self.allocation,
                _deltaP=0.05,
                _numOfusers=self.num_of_users,
                _RBTD=100,
                _Tstep=1,
                _Plot=False,
                _verbose=True,
                _unusual={"unusual": False, 'day': 10, "read": 'Trinitytt.npy', 'ctrlprice': 2.63,
                          'regulatesTime': 415, 'regulateeTime': 557, "dropstime": 360, "dropetime": 480,
                          "FTC": 0.05, "AR": 0.0},
                _storeTT={"flag": False, "ttfilename": 'Trinitylumptt'},
                _CV=False,
                save_dfname='./output/RL_Bottleneck/Trinity',
                toll_type=self.toll_type,
                _choiceInterval=self.choice_interval
            )
        sim.toll_parameter = np.array([self.toll_A, self.toll_mu, self.toll_sigma])
        sim.users.toll_parameter = np.array([self.toll_A, self.toll_mu, self.toll_sigma])
        return sim

    def _simulate_day_and_get_observation_info(self):
        tt_state, accumulation_state, sell_state, buy_state, pt_share_number, market_price, sw, tt_util, sde_util, sdl_util, ptwaiting_util, I_util, userBuy_util, userSell_util, fuelcost_util = self.sim.RL_simulateOneday(
            self.day, 5, self.state_shape)

        self.tt_interval_one_eps[self.day + 1] = tt_state

        observation = {
            "accumulation": np.array(accumulation_state, dtype=np.float64),
            "current_mu": np.array([self.toll_mu], dtype=np.float64),
            "current_sigma": np.array([self.toll_sigma], dtype=np.float64),
            "price": np.array([market_price], dtype=np.float64),
        }

        AITT_daily = np.mean(self.sim.flow_array[self.day, :, 2])
        tmp = self.sim.flow_array[self.day]
        tmp_2 = tmp[tmp[:, 0] != -1]
        AITT_car_only = np.mean(tmp_2[:, 2])

        reward = self._calculate_reward(AITT_daily, pt_share_number)

        self.last_AITT_daily = AITT_daily

        self._record_step_data(AITT_daily, sw, market_price, pt_share_number, reward)

        info = {
            "toll_parameter_mu": self.sim.toll_parameter[1],
            "toll_parameter_sigma": self.sim.toll_parameter[2],
            "rw": reward,
            "pt_share_number": pt_share_number,
            "sw": sw,
            "market_price": market_price,
            "AITT_daily": AITT_daily,
            "AITT_car_only": AITT_car_only,
        }

        done = self._is_last_day_of_episode()
        return observation, reward, done, info
