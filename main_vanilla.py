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
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import seaborn as sns


import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import NormalizeObservation

import CustomEnvs

def lrsched():
  def reallr(progress):
    lr = 0.001
    if progress < 0.80:
      lr = 0.0001
    return lr
  return reallr

def main():
    # todo
    train = True
    test = False
    resume = True
    one_dim = False
    best_toll_initialization = False
    reward_shifting = False
    train_episode = 600
    train_episode = train_episode+1
    simulation_day_num = 30 # the simulation days in one iteration 
    train_time_steps = int(simulation_day_num * train_episode)# total train times
    evaluation_time_episode = 20 # evaluation times
    checkpoint_save_episode = 30
    save_episode_train = 50 # the episode number to save train results
    save_episode_test = evaluation_time_episode 
    learning_rate = 0.001
    env_name = 'Custom_Envs/CommuteEnv-v4'
    if resume:
        resume_path = "PPO_Thu Sep 14 11:37:09 2023"
        resume_count = " seventh run "
        resume_step = 34200
    # finish todo 
        
    if one_dim:
        print(" one dim action ")
    else:
        print(" 3 dim actions")

    if best_toll_initialization:
        print(" Initialize with best toll  ")
    if reward_shifting:
        " shift reward based on average historic performance "
    print(" env: ", env_name)
    print(" learning_rate ", learning_rate)
    if resume: 
        resume_record = " resume from: " + resume_count + " "+resume_path + " "+str(resume_step) + "_steps"
        print(resume_record)
    if train: 
        save_dir = "./results/PPO_"+time.asctime(time.localtime(start_time))
        isExist = os.path.exists(save_dir)
        if not isExist:
            os.makedirs(save_dir)
        print(" Start training with total steps: ", train_time_steps,)
        env = gym.make(env_name,
                simulation_day_num = simulation_day_num, 
                save_episode_freq = save_episode_train, 
                train=True, save_dir = save_dir, space_shape=(4, int(12*60/5)),
                One_dim = one_dim, 
                Best_toll_initialization = best_toll_initialization, 
                Reward_shifting = reward_shifting
            )
        env = NormalizeObservation(env)
        checkpoint_callback = CheckpointCallback(save_freq= int(checkpoint_save_episode*simulation_day_num), 
                                                 save_path=(save_dir+"/logs/"), name_prefix="PPO")
        if resume == True: 
            model_path = "./results/" + resume_path +"/logs/PPO_"+str(resume_step)+"_steps"
            model = PPO.load(model_path, print_system_info=True, env=env)
            model.learn(total_timesteps = train_time_steps, callback = checkpoint_callback, reset_num_timesteps=False, tb_log_name= resume_count)
            model.save(save_dir+"/logs/"+str(train_time_steps)+"_steps")
        else:
            model = PPO("MlpPolicy", env, learning_rate=learning_rate, n_steps=150, verbose=1, batch_size=30, 
                        target_kl=0.05, n_epochs=40, gae_lambda=0.97,  ent_coef = 0.5, clip_range=0.2, gamma=0.99, 
                        tensorboard_log=(save_dir+"/tensorboards/PPO/"))
            model.learn(total_timesteps = train_time_steps, tb_log_name= "first_run", callback = checkpoint_callback)
            model.save(save_dir+"/logs/"+str(train_time_steps)+"_steps")
            print(model.learn)
        # save_freq is the save frequency steps
        print(" ")
        print("finish training!!!!!!!!!!!!!!!!!")
        print(" ")

    if test: 
        print(" Start testing with total eps: ", evaluation_time_episode)
        env = gym.make(env_name, simulation_day_num = simulation_day_num, 
                        save_episode_freq = save_episode_test, train=False, 
                        save_dir = save_dir, 
                        space_shape=(4, int(12*60/5)),
                        One_dim = one_dim, 
                        Best_toll_initialization = best_toll_initialization, 
                        Reward_shifting = reward_shifting
                        )
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