from collections import namedtuple
import os, time
from stable_baselines3.common.env_checker import check_env
from time import sleep
from collections import namedtuple
import os, time
import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation
import Custom_Envs
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3 import PPO, DDPG, TD3
import numpy as np

def lrsched():
  def reallr(progress):
    lr = 0.001
    if progress < 0.80:
      lr = 0.0001
    return lr
  return reallr

def main():
# todo starts
    one_dim = True
    best_toll_initialization = False
    reward_shifting = True
    random_mode = True
    # training parameters
    env_name = 'Custom_Envs/CommuteEnv-v4'
    evaluation_time_episode = 100 # evaluation times
    # test parameters
# todo finish 

    save_episode_test = evaluation_time_episode 
    simulation_day_num = 30 # the simulation days in one iteration 
    print(" running random policy ")
    if one_dim:
        print(" one dim action ")
    else:
        print(" 3 dim actions")
    if random_mode:
        print(" random policy  ")
    if best_toll_initialization:
        print(" Initialize with best toll  ")
    if reward_shifting:
        " shift reward based on average historic performance "
    
    start_time = time.time()
    save_dir = "./results/PPO_"+time.asctime(time.localtime(start_time))
    print(" save_dir ", save_dir)
    isExist = os.path.exists(save_dir)
    if not isExist:
        os.makedirs(save_dir)

    # environment initialization
    env = gym.make(env_name, simulation_day_num = simulation_day_num, 
                    save_episode_freq = save_episode_test, 
                    train=False, save_dir = save_dir, space_shape=(4, int(12*60/5)),
                    One_dim = one_dim, 
                    Best_toll_initialization = best_toll_initialization, 
                    Reward_shifting = reward_shifting)
    env = NormalizeObservation(env)

    for _ in range(evaluation_time_episode):
        done  = False
        obs = env.reset()[0] 
        while not done:  # every step has same seed
            action = [env.action_space.sample()]
            print("action ", action)
            obs, reward, terminated, done, info = env.step(action[0])
        
if __name__ == "__main__":
    start_time = time.time()
    print("start_time ", time.asctime(time.localtime(start_time)))

    main()

    end_time = time.time()
    print("end_time ",time.asctime(time.localtime(end_time)))
    print("total elapsed time ",end_time-start_time)