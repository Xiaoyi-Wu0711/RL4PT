
import numpy as np
from numba import njit, prange
import pandas as pd
import argparse
import pickle
from collections import namedtuple
from itertools import count
import random
import os, time
import numpy as np
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from time import sleep
import datetime
import argparse
import pickle
from collections import namedtuple
from itertools import count
import random
import os, time
import numpy as np
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from time import sleep
import datetime
import random
import numpy as np
import pandas as pd
from numba import njit, prange
from gym import Env 
from gym import spaces 
#IMPORTS
import random
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from simulation import CommuteEnv

def main():
    train = True
    test = True
    resume = False 
    train_episode = 300
    simulation_day_num = 30 # the simulation days in one iteration 
    train_time_steps = int(simulation_day_num * train_episode)# total train times
    evaluation_time_episode = 30 # evaluation times
    save_episode_train = 10 # the episode number to save train results
    save_episode_test = evaluation_time_episode
    learning_rate = 0.01
    print(" simulation is simulation 1 " )

    print(" learning_rate ", learning_rate)
    if train: 
        save_dir = "./results/PPO_"+time.asctime(time.localtime(start_time))
        isExist = os.path.exists(save_dir)
        if not isExist:
            os.makedirs(save_dir)
        print(" Start training with total steps: ", train_time_steps)
        env = CommuteEnv(simulation_day_num = simulation_day_num, save_episode_freq = save_episode_train, train = True, save_dir = save_dir, space_shape = (4, int(12*60/5)))
        seed_value = 0
        checkpoint_callback = CheckpointCallback(save_freq= int(save_episode_train*30), save_path=(save_dir+"/logs/"), name_prefix="PPO")
        if resume == True: 
            model = PPO.load(("./results/PPO_Wed Aug 16 15:16:15 2023/logs/PPO_"+str(900)+"_steps"), print_system_info=True, env=env)
            model.learn(total_timesteps = train_time_steps, callback = checkpoint_callback, reset_num_timesteps=False, tb_log_name= "second_run")
            model.save(save_dir+"/logs/"+str(900+train_time_steps)+"_steps")
        else:
            model = PPO("MlpPolicy", env, learning_rate=learning_rate, n_steps=150, verbose=1, batch_size=30, 
                    target_kl=0.05, n_epochs=40, gae_lambda=0.97, tensorboard_log=(save_dir+"/tensorboards/PPO/"))
            model.learn(total_timesteps = train_time_steps, callback = checkpoint_callback)
            model.save(save_dir+"/logs/"+str(train_time_steps)+"_steps")
        # save_freq is the save frequency steps
        print(" ")
        print("finish training!!!!!!!!!!!!!!!!!")
        print(" ")

    if test: 
        print(" Start testing with total eps: ", evaluation_time_episode)
        simulation_day_num = simulation_day_num
        env = CommuteEnv(simulation_day_num = simulation_day_num, save_episode_freq = save_episode_test, train = False, save_dir = save_dir, space_shape = (4, int(12*60/5)))
        for i in range(evaluation_time_episode):
            seed_value = random.randint(0, 99999999)
            print(seed_value)
            env.set_seed(seed_value) #every episode need to reset env and reset the seed
            done  = False
            obs = env.reset() 
            while not done:  # every step has same seed
                action = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action[0])
        print("finish test¡ng!!!!!!!!!!!!!!!!!")
        
if __name__ == "__main__":
    start_time = time.time()
    print("start_time ", time.asctime(time.localtime(start_time)))

    main()

    end_time = time.time()
    print("end_time ",time.asctime(time.localtime(end_time)))
    print("total elapsed time ",end_time-start_time)