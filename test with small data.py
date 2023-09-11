from collections import namedtuple
from itertools import count
import random
import os, time
import numpy as np
import Custom_Envs
import torch
import torch.nn as nn
from stable_baselines3.common.env_checker import check_env
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
import seaborn as sns
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback


import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import NormalizeObservation

import CustomEnvs

def lrsched():
  def reallr(progress):
    lr = 0.1
    if progress < 0.85:
      lr = 0.05
    if progress < 0.66:
      lr = 0.01
    if progress < 0.33:
      lr = 0.001
    return lr
  return reallr
print(" test pipeine")
def main():
    train = True
    test = True
    resume = False 
    train_episode = 5
    simulation_day_num = 2 # the simulation days in one iteration 
    train_time_steps = int(simulation_day_num * train_episode)# total train times
    evaluation_time_episode = 2 # evaluation times
    save_episode_train = 2 # the episode number to save train results
    save_episode_test = evaluation_time_episode 
    learning_rate = 0.001
    # print(" linear reduction learning rate ")
    print(" learning_rate ", learning_rate)

    if train: 
        save_dir = "./results/PPO_"+time.asctime(time.localtime(start_time))
        isExist = os.path.exists(save_dir)
        if not isExist:
            os.makedirs(save_dir)
        print(" Start training with total steps: ", train_time_steps,)
        env = gym.make('Custom_Envs/CommuteEnv-v0', simulation_day_num = simulation_day_num, 
                        save_episode_freq = save_episode_train, 
                        train=True, save_dir = save_dir, space_shape=(4, int(12*60/5)))
        env = NormalizeObservation(env)
        checkpoint_callback = CheckpointCallback(save_freq= int(save_episode_train*simulation_day_num), 
                                                 save_path=(save_dir+"/logs/"), name_prefix="PPO")
        # when we do the evaluation actually every time we need to reset the env
        eval_env =  gym.make('Custom_Envs/CommuteEnv-v0', simulation_day_num = simulation_day_num, 
                        save_episode_freq = evaluation_time_episode, 
                        train=False, save_dir = save_dir, space_shape=(4, int(12*60/5)))
        eval_env = NormalizeObservation(eval_env)
        eval_callback = EvalCallback(eval_env, best_model_save_path= (save_dir+"/best_model"), 
                                     log_path=(save_dir+"/best_model/"), verbose=1,
                                     eval_freq= int(evaluation_time_episode))
        
        callback_ls = CallbackList([checkpoint_callback, eval_callback])

        if resume == True: 
            model = PPO.load(("./results/PPO_Sun Sep  3 14:19:04 2023/logs/PPO_"+str(5400)+"_steps"), 
                             print_system_info=True, env=env)
            model.learn(total_timesteps = train_time_steps, callback = callback_ls, 
                        reset_num_timesteps=False, tb_log_name= "second_run")
            model.save(save_dir+"/logs/"+str(train_time_steps)+"_steps")

        else:
            model = PPO("MlpPolicy", env, learning_rate=learning_rate, n_steps=5, verbose=1, 
                        batch_size=4, target_kl=0.05, n_epochs=2, 
                        gae_lambda=0.97, tensorboard_log=(save_dir+"/tensorboards/PPO/"))
            model.learn(total_timesteps = train_time_steps, tb_log_name= "first_run", 
                        callback = callback_ls)
            model.save(save_dir+"/logs/"+str(train_time_steps)+"_steps")
            print(model.learn)
        # save_freq is the save frequency steps
        print(" ")
        print("finish training!!!!!!!!!!!!!!!!!")
        print(" ")

    if test: 
        print(" Start testing with total eps: ", evaluation_time_episode)
        env = gym.make('Custom_Envs/CommuteEnv-v0', simulation_day_num = simulation_day_num, 
                        save_episode_freq = save_episode_test, train=False, save_dir = save_dir, 
                        space_shape=(4, int(12*60/5)))
        env = NormalizeObservation(env)        
        for i in range(evaluation_time_episode):
            done  = False
            obs = env.reset()[0] 
            while not done:  # every step has same seed
                action = model.predict(obs, deterministic=True)
                obs, reward, terminated, done, info = env.step(action[0])
        print("finish test¡ng!!!!!!!!!!!!!!!!!")

        
if __name__ == "__main__":
    start_time = time.time()
    print("start_time ", time.asctime(time.localtime(start_time)))

    main()

    end_time = time.time()
    print("end_time ",time.asctime(time.localtime(end_time)))
    print("total elapsed time ",end_time-start_time)