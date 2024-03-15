import os, time
import Custom_Envs
import numpy as np
import os, time

from helper import SaveVecNormalizeCallback, lrsched, TensorboardCallback

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import shutil

import multiprocessing
from multiprocessing import Pool

absolute_change_mode = False
initialization = "NT" # NT, random, best
toll_type = "normal" # normal or step toll type
supply_model = "MFD" # Bottleneck or MFD
capacity = 7000
state_shape = (5, int(12*60/5))
action_shape = (1, )
env_id = 'Custom_Envs/CommuteEnv-v5' 
choiceInterval = 60
allocation =  {'AR': 0.00269, 'way': 'continuous',
                'FTCs': 0.05, 'FTCb': 0.05, 
                'PTCs': 0.00, 'PTCb': 0.00,
                "Decaying": False}


##### todo start ###########
# n_envs = 1
# train_episode = n_envs * 1 # each env train 5 episodes
# evaluation_time_episode = 1 # each evaluation has xx episode 
# training_n_epochs = 2 # each updates have training_n_epochs epochs 
# simulation_day_num = 2  # the simulation days in one iteration 
# eval_freq_episode = 1 # for each env, every 2 episode do once evaluation
# checkpoint_save_episode = n_envs   # this is the checkpoint for xx episodes, save model for checkpoint_save_episode * simulation_days
# training_n_episode = 2  # The number of episode to run for each environment per update
# batch_size_episode = n_envs # episode length in batch size 

n_envs = 10
train_episode = n_envs * 100 # each env train 5 episodes
evaluation_time_episode = 1 # each evaluation has xx episode 
training_n_epochs = 10 # each updates have training_n_epochs epochs 
simulation_day_num = 60  # the simulation days in one iteration 
eval_freq_episode = 1 # for each env, every 2 episode do once evaluation
checkpoint_save_episode = n_envs  # this is the checkpoint for xx episodes, save model for checkpoint_save_episode * simulation_days
training_n_episode = 1  # The number of episode to run for each environment per update
batch_size_episode = n_envs # episode length in batch size 

policy_kwargs =  {"net_arch" :[16, 16]}
action_weights = 2
std_weights = 0.5
device = "cpu"
resume = False
if resume:
    resume_path = "PPO_Mon Feb  5 19:57:20 2024"
    resume_count = " second run "
    resume_step = 30000
####### todo finish ###########

# RL hyperparameter
training_n_steps = simulation_day_num * training_n_episode # each n steps update for each env
training_entropy_coef = 0.2
training_learning_rate = 1e-3
training_gae_lambda = 0.97
training_clip_range = 0.2
training_gamma = 1
training_target_kl = 0.05
reward_scheme = "fftt" # fftt, Weighted_reward, Add_constant, Shifting
reward_weight = 100
training_batch_size = int(simulation_day_num * batch_size_episode)
# train_episode += 1
train_time_steps = int(simulation_day_num * train_episode)# total train times
eval_freq = eval_freq_episode * simulation_day_num

def make_env_train(env_id, i):
    save_dir = train_dir+"env_"+str(i)+"/"
    isExist = os.path.exists(save_dir)
    if not isExist:
        os.makedirs(save_dir)

    def _init():
        env =  gym.make(env_id, 
                        simulation_day_num = simulation_day_num, 
                        save_episode_freq = checkpoint_save_episode, 
                        save_dir = save_dir, 
                        state_shape = state_shape,
                        action_shape = action_shape,
                        Initialization = initialization, 
                        Absolute_change_mode = absolute_change_mode,
                        Reward_scheme = reward_scheme, 
                        reward_weight = reward_weight,
                        Toll_type = toll_type,
                        Supply_model = supply_model, 
                        Mode =  "train", 
                        episode_in_one_eval = evaluation_time_episode,
                        Allocation = allocation,
                        input_save_dir = input_dir,
                        action_weights = action_weights,
                        std_weights = std_weights
                    )
        env = Monitor(env, 
                      filename = train_dir + "monitor.csv", 
                      info_keywords = ("sw", "pt_share_number", "market_price", "AITT_daily") 
                    )
        env.reset()
        # env.seed(seed)
        return env
    # env = env.unwrapped
    return _init


def make_env_eval(env_id, i):
    save_dir = eval_dir +"env_"+str(i)+"/"
    isExist = os.path.exists(save_dir)
    if not isExist:
        os.makedirs(save_dir)
    def _init():
        env =  gym.make(env_id, 
                        simulation_day_num = simulation_day_num, 
                        save_episode_freq = checkpoint_save_episode, 
                        save_dir = save_dir, 
                        state_shape = state_shape,
                        action_shape = action_shape,
                        Initialization = initialization, 
                        Absolute_change_mode = absolute_change_mode,
                        Reward_scheme = reward_scheme, 
                        reward_weight = reward_weight,
                        Toll_type = toll_type,
                        Supply_model = supply_model, 
                        Mode =  "eval", 
                        episode_in_one_eval = evaluation_time_episode,
                        Allocation = allocation,
                        input_save_dir = input_dir,
                        action_weights = action_weights,
                        std_weights = std_weights

                    )
            
        env = Monitor(env, 
                      filename = eval_dir + "monitor.csv", 
                      info_keywords = ("sw", "pt_share_number", "market_price", "AITT_daily") 
                      )

        env.reset()
        # env.seed(seed)
        return env
    # env = env.unwrapped
    return _init

if __name__ == "__main__":

    if resume: 
        resume_record = "Resume from: " + resume_count + " "+resume_path + " "+str(resume_step) + "_steps"
        print(resume_record)

    if action_shape[0] == 1:
        print("Action: 1 dim action(A) ")
    
    elif action_shape[0] == 2:
        print("Action: 1 dim action(A, mu) ")
    else:
        print("Action: 3 dim actions(A, sigma, mu)")
    
    print("PT TT is 60 minute")
    if absolute_change_mode==True:
        print("Action: Absolute value of toll parameters")
    else:
        print("Action: Relative value change of toll parameters")

    print("Initialize toll: ", initialization)
    print("Choice interval ", choiceInterval)
    print("Allocation rate ", allocation["AR"])
    
    print("action_weights ", action_weights)
    print("std_weights ", std_weights)


    if reward_scheme == "Shifting":
        print("Reward: AITT_t' - AITT_t ")
    elif reward_scheme == "Add_constant":
        if supply_model == "Bottleneck":
            print("Reward: -AITT + 30 ")
        else: 
            print("Reward: -AITT + 45 ")
    elif reward_scheme == "Weighted_reward":
        print("Reward: (-AITT/(reward_weight*FFTT)", str(reward_weight))
    else:
        print("Reward:  -AITT/FFTT")

    print("State: ", state_shape)
    print("Supply model: ", supply_model)
    print("Step number in one episode: ", simulation_day_num)
    print("Capacity: ", capacity)
    print("train_time_steps: ", train_time_steps)
    print("training_entropy_coef: ", training_entropy_coef)
    print("training_learning_rate: ", training_learning_rate)
    print("training_gae_lambda: ", training_gae_lambda)
    print("training_clip_range: ", training_clip_range)
    print("training_gamma: ", training_gamma)
    print("training_target_kl: ", training_target_kl)
    print("device in training: ", device)
    print("start_time: ", time.asctime(time.localtime(time.time())))

    print("--------------------------------------------------")
    
    start_time = time.time()
    save_dir = "./results_4/RL_"+supply_model+"_"+toll_type+"/PPO_"+time.asctime(time.localtime(start_time))
    train_dir = save_dir + "/train/"
    isExist = os.path.exists(train_dir)
    if not isExist:
        os.makedirs(train_dir)

    eval_dir = save_dir + "/eval/"
    isExist = os.path.exists(eval_dir)
    if not isExist:
        os.makedirs(eval_dir)

    input_dir = "./output/MFD_pt_60_changed_ratio/NT/"


    train_env = SubprocVecEnv([make_env_train(env_id, i) for i in range(n_envs)], start_method="forkserver")
    train_env = VecNormalize(train_env, 
                             training = True,
                             norm_obs = True, 
                             norm_reward = True, 
                             gamma = training_gamma,
                        )
    
    eval_env = SubprocVecEnv([make_env_eval(env_id, i) for i in range(1)], start_method="forkserver")
    eval_env = VecNormalize(eval_env, 
                            training = False,
                            norm_obs = True, 
                            norm_reward = False, 
                            gamma = training_gamma,
                        )
    
    save_vec_normalize_callback = SaveVecNormalizeCallback(save_path=save_dir + "/eval_best_model")
    eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path = save_dir + "/eval_best_model",
                                 log_path = eval_dir, 
                                 eval_freq = eval_freq, 
                                 n_eval_episodes = evaluation_time_episode,
                                 deterministic = True,
                                 render = False,
                                 callback_on_new_best = save_vec_normalize_callback
                            )
    checkpoint_callback = CheckpointCallback(save_freq= int(checkpoint_save_episode*simulation_day_num), 
                                             save_path = (save_dir+"/logs/"), 
                                             name_prefix = "PPO",
                                             save_vecnormalize = True)
    callback_ls = CallbackList([eval_callback, checkpoint_callback, TensorboardCallback()])

    # it is recommended to run several experiments due to variability in results
    train_env.reset()
    start = time.time()
    if resume == True: 
            model_path = "./results_4/RL_MFD_normal/" + resume_path +"/logs/PPO_"+str(resume_step)+"_steps"
            model = PPO.load(model_path, 
                             print_system_info = True, 
                             env = train_env
                            )
            model.learn(total_timesteps = train_time_steps, 
                        callback = callback_ls, 
                        reset_num_timesteps = False, 
                        tb_log_name = resume_count, 
                        )
            model.save(save_dir+"/logs/"+str(train_time_steps)+"_steps")
    else:
        model = PPO("MultiInputPolicy", 
                train_env, 
                policy_kwargs = policy_kwargs,
                learning_rate = lrsched(), 
                n_steps = training_n_steps, 
                verbose = 1,
                batch_size = training_batch_size, 
                target_kl = training_target_kl, 
                n_epochs = training_n_epochs, 
                gae_lambda = training_gae_lambda,  
                ent_coef = training_entropy_coef, 
                clip_range = training_clip_range, 
                gamma = training_gamma, 
                tensorboard_log = (train_dir+"tensorboards/PPO/"),
                device = device
            )
        print(model.policy)
        model.learn(total_timesteps = train_time_steps,
                    tb_log_name = "first_run", 
                    callback = callback_ls)
    model.save(save_dir+"/logs/last_model.zip")
    print(model.learn)
    train_env.close()
    end = time.time()

    print("total time is: ", end - start)
    print("finished")