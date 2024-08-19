import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from helper import SaveVecNormalizeCallback, lrsched, TensorboardCallback
import shutil
import Custom_Envs

# Configuration Parameters
absolute_change_mode = False
initialization = "NT"  # NT, random, best
toll_type = "normal"  # normal or step toll type
supply_model = "MFD"  # Bottleneck or MFD
state_shape = (5, int(12 * 60 / 5))
choiceInterval = 60
allocation = {
    'AR': 0.00269, 'way': 'continuous',
    'FTCs': 0.05, 'FTCb': 0.05,
    'PTCs': 0.00, 'PTCb': 0.00,
    "Decaying": False
}
capacity = int(7000)
n_envs = 10
train_episode = n_envs * 100
evaluation_time_episode = 1 
training_n_epochs = 10 # update time for using whole batch size
simulation_day_num = 60
checkpoint_save_episode = 1
training_n_episode = 4 # episode number each env collects for roll-out
eval_freq_episode = training_n_episode 
batch_size_episode = n_envs # batch size 

std_weights = 0.5
pt_weights = 1
device = "cpu"
resume = False

if resume:
    resume_path = "PPO_Mon May  6 09:22:09 2024"
    resume_count = " second run "
    resume_step = 12000

training_entropy_coef = 0.2
training_learning_rate = "decayed function"
training_gae_lambda = 1
training_clip_range = 0.2
training_gamma = 1
training_target_kl = 0.05
# reward_scheme = "triangle_add_constant"
reward_scheme = "triangle"
reward_weight = 50
numOfusers = int(7500)
training_log_std_init = 0
seed_value = 111

policy_kwargs = {
    # "net_arch": dict(pi=[4,4], vf=[4,4]),
    # "net_arch": [8, 8],    
    # "net_arch": [8, 16, 16, 8],
    "net_arch": [8, 16],

    "log_std_init": training_log_std_init
}

env_id = 'Custom_Envs/CommuteEnv-v8'

# relative act and rw
model_parameter = "capacity"
input_dir = "output/MFD/"+str(model_parameter) + "/"+str(capacity) + "/NT/" 

# todo finish 
train_time_steps = int(simulation_day_num * train_episode)
eval_freq = int(eval_freq_episode * simulation_day_num)
training_batch_size = int(simulation_day_num * batch_size_episode)
training_n_steps = int(simulation_day_num * training_n_episode)

if env_id == 'Custom_Envs/CommuteEnv-v5': # A
    action_shape, action_weights = (1,), (2,)
elif env_id == 'Custom_Envs/CommuteEnv-v6': # mu
    action_shape, action_weights = (1,), (5,)
elif env_id == 'Custom_Envs/CommuteEnv-v7': # sigma
    action_shape, action_weights = (1,), (3,)
elif env_id == 'Custom_Envs/CommuteEnv-v8': # A and mu
    action_shape, action_weights = (2,), (2, 15)
elif env_id == 'Custom_Envs/CommuteEnv-v9': # A and sigma
    action_shape, action_weights = (2,), (2, 3) 
elif env_id == 'Custom_Envs/CommuteEnv-v10': # mu and sigma
    action_shape, action_weights = (2,), (10, 3)
elif env_id == 'Custom_Envs/CommuteEnv-v11': # A, mu, and sigma, 
    action_shape, action_weights = (3,), (2, 15, 3)

def make_env(env_id, i, mode, save_dir, input_dir):
    def _init():
        env = gym.make(
            env_id,
            input_save_dir=input_dir, # input dir
            save_dir=save_dir, # output dir
            episode_in_one_eval=evaluation_time_episode, # evaluation eposide number
            save_episode_freq=checkpoint_save_episode,
            state_shape=state_shape, # RL hyperparameters
            action_shape=action_shape,
            Initialization=initialization,
            Absolute_change_mode=absolute_change_mode,
            Reward_scheme=reward_scheme,
            reward_weight=reward_weight,
            action_weights=action_weights,
            simulation_day_num=simulation_day_num,# Env hyperparameters
            Toll_type=toll_type,
            Supply_model=supply_model,
            Mode=mode,
            Allocation=allocation,
            std_weights=std_weights,
            pt_weights=pt_weights,
            numOfusers=numOfusers
        )
        env = Monitor(env, filename=f"{save_dir}/monitor.csv",
                      info_keywords=("sw", "pt_share_number", "market_price", "AITT_daily"))
        env.reset()
        return env
    return _init

def create_directories(*dirs):
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)

def configure_hyperparameters():
    hyperparams = {
        "absolute_change_mode": absolute_change_mode,
        "initialization": initialization,
        "toll_type": toll_type,
        "supply_model": supply_model,
        "state_shape": state_shape,
        "choiceInterval": choiceInterval,
        "allocation": allocation,
        "capacity": capacity,
        "n_envs": n_envs,
        "train_episode": train_episode,
        "evaluation_time_episode": evaluation_time_episode,
        "training_n_epochs": training_n_epochs,
        "simulation_day_num": simulation_day_num,
        "checkpoint_save_episode": checkpoint_save_episode,
        "training_n_episode": training_n_episode,
        "eval_freq_episode": eval_freq_episode,
        "batch_size_episode": batch_size_episode,
        "std_weights": std_weights,
        "pt_weights": pt_weights,
        "device": device,
        "resume": resume,
        "training_n_steps": training_n_steps,
        "training_entropy_coef": training_entropy_coef,
        "training_learning_rate": training_learning_rate,
        "training_gae_lambda": training_gae_lambda,
        "training_clip_range": training_clip_range,
        "training_gamma": training_gamma,
        "training_target_kl": training_target_kl,
        "reward_scheme": reward_scheme,
        "reward_weight": reward_weight,
        "numOfusers": numOfusers,
        "training_batch_size": training_batch_size,
        "training_log_std_init": training_log_std_init,
        "train_time_steps": train_time_steps,
        "eval_freq": eval_freq,
        "policy_kwargs": policy_kwargs,
        "env_id": env_id,
        "action_shape": action_shape,
        "action_weights": action_weights,
        "input_dir": input_dir,
        "seed_value": seed_value,
    }
    return hyperparams

def log_hyperparameters(hyperparams):
    print("Hyperparameters:")
    for key, value in hyperparams.items():
        print(f"{key}: {value}")
    print("--------------------------------------------------")

if __name__ == "__main__":

    if resume:
        resume_record = f"Resume from: {resume_count} {resume_path} {resume_step}_steps"
        print(resume_record)

    hyperparams = configure_hyperparameters()
    log_hyperparameters(hyperparams)

    os.chdir('RL4PT/')
    current_working_directory = os.getcwd()
    print(f"Current working directory: {current_working_directory}")

    start_time = time.time()
    save_dir = f"./results_4/RL_{supply_model}_{toll_type}/PPO_{time.asctime(time.localtime(start_time))}"
    train_dir = f"{save_dir}/train/"
    eval_dir = f"{save_dir}/eval/"

    create_directories(train_dir, eval_dir)

    train_env = SubprocVecEnv([make_env(env_id, i, "train", train_dir+"env_"+str(i)+"/", input_dir) for i in range(n_envs)], start_method="forkserver")
    train_env = VecNormalize(train_env, training=True, norm_obs=True, norm_reward=False, gamma=training_gamma)
    
    eval_env = SubprocVecEnv([make_env(env_id, i, "eval",  eval_dir +"env_"+str(i)+"/", input_dir) for i in range(1)], start_method="forkserver")
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, gamma=training_gamma)
    
    save_vec_normalize_callback = SaveVecNormalizeCallback(save_path=f"{save_dir}/eval_best_model")
    eval_callback = EvalCallback(eval_env, best_model_save_path=f"{save_dir}/eval_best_model", log_path=eval_dir,
                                 eval_freq=eval_freq, n_eval_episodes=evaluation_time_episode, deterministic=True,
                                 render=False, callback_on_new_best=save_vec_normalize_callback)
    checkpoint_callback = CheckpointCallback(save_freq=int(checkpoint_save_episode * simulation_day_num),
                                             save_path=f"{save_dir}/logs/", name_prefix="PPO", save_vecnormalize=True)
    callback_ls = CallbackList([eval_callback, checkpoint_callback, TensorboardCallback(save_dir=save_dir)])

    train_env.reset()
    start = time.time()
    try:
        if resume:
            model_path = f"./results_4/RL_MFD_normal/{resume_path}/logs/PPO_{resume_step}_steps"
            model = PPO.load(model_path, print_system_info=True, env=train_env)
            model.learn(total_timesteps=train_time_steps, callback=callback_ls, reset_num_timesteps=False,
                        tb_log_name=resume_count)
            model.save(f"{save_dir}/logs/{train_time_steps}_steps")
        else:
            model = PPO("MultiInputPolicy", train_env, policy_kwargs=policy_kwargs, learning_rate=lrsched(), 
                        n_steps=training_n_steps, verbose=1, batch_size=training_batch_size, target_kl=training_target_kl, 
                        n_epochs=training_n_epochs, gae_lambda=training_gae_lambda, ent_coef=training_entropy_coef, 
                        clip_range=training_clip_range, gamma=training_gamma, tensorboard_log=f"{train_dir}/tensorboards/PPO/", 
                        device=device, seed = seed_value)
            print(model.policy)
            model.learn(total_timesteps=train_time_steps, tb_log_name="first_run", callback=callback_ls)
            model.save(f"{save_dir}/logs/last_model.zip")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        train_env.close()
        end = time.time()
        print(f"Total time is: {end - start}")
        print("Finished")
