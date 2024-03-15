import os, time
import Custom_Envs

import os, time

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation
from customer_tensorboard_Callback import TensorboardCallback, TimeLimitWrapper
import CustomEnvs
from stable_baselines3.common.monitor import Monitor
import numpy as np
import multiprocessing
from multiprocessing import Pool

import CustomEnvs

def lrsched():
  def reallr(progress):
    lr = 0.001
    if progress < 0.80:
      lr = 0.0001
    return lr
  return reallr

action_ls_from_RL = [0.        , 0.62485766, 1.35882157, 2.63746041, 3.9007197 ,
       4.81398231, 4.62268466, 4.95904222, 4.68808609, 4.77824273,
       4.41579881, 4.36716848, 4.47026377, 4.03110223, 3.7344503 ,
       3.92708493, 3.96195883, 3.96951514, 4.10824616, 3.89661486,
       3.80133263, 3.9222701 , 3.92215859, 4.32351499, 3.86652081,
       3.67871796, 4.18331813, 3.78294044, 3.87381356, 4.00613096,
       3.69773113, 3.49604278, 4.23639583, 3.76318198, 3.91378022,
       3.79494537, 4.0083484 , 3.80811691, 4.07072049, 4.03574235,
       4.32367504, 3.87118245, 3.83316778, 4.19787519, 3.83245151,
       3.89068998, 3.81917504, 4.47622196, 4.0476282 , 4.09396099,
       3.89489814, 3.83705863, 4.24289336, 3.7882727 , 3.73189237,
       4.36990204, 3.8688946 , 4.0185326 , 3.59422537, 4.14950081,
       3.5851697 ]

def Simulate(action):
   # todo --------------------------------------------------------
    absolute_change_mode = False
    initialization = "NT" # NT, random, best
    simulation_day_num = 60
    toll_type = "normal" # normal or step toll type
    supply_model = "MFD" # Bottleneck or MFD
    capacity = 7000
    state_shape = (5, int(12*60/5))
    action_shape = (1, )
    env_name = 'Custom_Envs/CommuteEnv-v4' 
    choiceInterval = 60
    allocation =  {'AR': 0.00269, 'way': 'continuous',
                   'FTCs': 0.05, 'FTCb': 0.05, 'PTCs': 0.00, 
                   'PTCb': 0.00,
                   "Decaying": False}
    reward_scheme = "fftt" # fftt, Weighted_reward, Add_constant, Shifting
    reward_weight = 100
    evaluation_time_episode  = 1
    checkpoint_save_episode = evaluation_time_episode
    action_weights = 2
    # finish todo --------------------------------------------------------

    rw_ls = [] 
    action_ls = [] 
    toll_parameter_A_ls = []
    AITT_daily_ls = []
    AITT_car_only_ls = []
    pt_ls = [] 
    mp_ls = []
    sw_ls = [] 
    tt_util_ls = []
    sde_util_ls = []
    sdl_util_ls = []
    ptwaiting_util_ls = []
    I_util_ls = []
    userBuy_util_ls = []
    userSell_util_ls = []
    fuelcost_util_ls = []
    
    print("set seed to 333")
    print("Action: 1 dim action(A): ", action )

    if action_shape[0] == 1:
        print("Action: 1 dim action(A) ")
    elif action_shape[0] == 2:
        print("Action: 1 dim action(A, mu) ")
    else:
        print("Action: 3 dim actions(A, sigma, mu)")
    
    if absolute_change_mode==True:
        print("Action: Absolute value of toll parameters")
    else:
        print("Action: Relative value change of toll parameters")

    print("Initialize toll: ", initialization)
    print("Choice interval ", choiceInterval)
    print("Allocation rate ", allocation["AR"])

    if reward_scheme == "Shifting":
        print("Reward: AITT_t' - AITT_t ")
    elif reward_scheme == "Add_constant":
        if supply_model == "Bottleneck":
            print("Reward: -AITT + 30 ")
        else: 
            print("Reward: -AITT + 45 ")
    elif reward_scheme == "Weighted_reward":
        print("Reward: (-AITT/(10*FFTT))*", str(reward_weight))
    else:
        print("Reward:  -AITT/FFTT")


    print("State: ", state_shape)
    print("Supply model: ", supply_model)
    print("Step number in one episode: ", simulation_day_num)
    print("Capacity: ", capacity)
    print("--------------------------------------------------")

    save_dir = "./results_NORL_pt_60_changed_ratio/A_std_0_5/"
    isExist = os.path.exists(save_dir)
    if not isExist:
        os.makedirs(save_dir)

    input_dir = "output/MFD_pt_60_changed_ratio/NT/"

    env = gym.make(env_name, 
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
                    Mode =  "simulation", 
                    episode_in_one_eval = evaluation_time_episode,
                    Allocation = allocation,
                    input_save_dir = input_dir,
                    action_weights = 2
                )
    env = Monitor(env, filename= save_dir+"monitor.csv", info_keywords=("sw", "pt_share_number", "market_price", "AITT_daily"))
    # env = NormalizeObservation(env) 

    for i in range(evaluation_time_episode):# every episode has different seed
        done  = False
        obs, info = env.reset() # get the first obs
        toll_parameter_A_ls.append(info["toll_parameter_A"])
        AITT_daily_ls.append(info["AITT_daily"])
        AITT_car_only_ls.append(info["AITT_car_only"])
        pt_ls.append(info["pt_share_number"])
        mp_ls.append(info["market_price"])
        sw_ls.append(info["sw"])
        tt_util_ls.append(info["tt_util"])
        sde_util_ls.append(info["sde_util"])
        sdl_util_ls.append(info["sdl_util"])
        ptwaiting_util_ls.append(info["ptwaiting_util"])
        I_util_ls.append(info["I_util"])
        userBuy_util_ls.append(info["userBuy_util"])
        userSell_util_ls.append(info["userSell_util"])
        fuelcost_util_ls.append(info["fuelcost_util"])
        count = 0 

        while not done:  # every step has different seed

            print(" count ", count)
            # print(" done ", done)
            if int(action) == 9:
                if count%2 == 0:
                    action_2 = 7
                else:
                    action_2 = 3

            elif int(action) == 10:
                if count%2 == 0:
                    action_2 = 3
                else:
                    action_2 = 7 

            elif int(action) == 11:
                action_2 = np.random.random()*7
            
            elif int(action) == 12:
                if count%2 == 0:
                    action_2 = 4
                else:
                    action_2 = 2
            
            elif int(action) == 13:
                if count%2 == 0:
                    action_2 = 2
                else:
                    action_2 = 4
            
            elif int(action)  == 14:
                action_2 = action_ls_from_RL[count]

            else:
                action_2 = action    
        
            print("action_2 ", action_2)

            obs, reward, done, terminated, info = env.step([action_2])

            toll_parameter_A_ls.append(info["toll_parameter_A"])
            AITT_daily_ls.append(info["AITT_daily"])
            AITT_car_only_ls.append(info["AITT_car_only"])
            pt_ls.append(info["pt_share_number"])
            mp_ls.append(info["market_price"])
            sw_ls.append(info["sw"])
            tt_util_ls.append(info["tt_util"])
            sde_util_ls.append(info["sde_util"])
            sdl_util_ls.append(info["sdl_util"])
            ptwaiting_util_ls.append(info["ptwaiting_util"])
            I_util_ls.append(info["I_util"])
            userBuy_util_ls.append(info["userBuy_util"])
            userSell_util_ls.append(info["userSell_util"])
            fuelcost_util_ls.append(info["fuelcost_util"])

            count += 1

    np.save(save_dir + "toll_parameter_A_ls.npy", np.array(toll_parameter_A_ls))
    np.save(save_dir + "AITT_daily_ls.npy", np.array(AITT_daily_ls))
    np.save(save_dir + "AITT_car_only_ls.npy", np.array(AITT_car_only_ls))
    np.save(save_dir + "pt_ls.npy", np.array(pt_ls))
    np.save(save_dir + "mp_ls.npy", np.array(mp_ls))
    np.save(save_dir + "sw_ls.npy", np.array(sw_ls))
    np.save(save_dir + "tt_util_ls.npy", np.array(tt_util_ls))
    np.save(save_dir + "sde_util_ls.npy", np.array(sde_util_ls))
    np.save(save_dir + "sdl_util_ls.npy", np.array(sdl_util_ls))
    np.save(save_dir + "ptwaiting_util_ls.npy", np.array(ptwaiting_util_ls))
    np.save(save_dir + "I_util_ls.npy", np.array(I_util_ls))
    np.save(save_dir + "userBuy_util_ls.npy", np.array(userBuy_util_ls))
    np.save(save_dir + "userSell_util_ls.npy", np.array(userSell_util_ls))
    np.save(save_dir + "fuelcost_util_ls.npy", np.array(fuelcost_util_ls))

    print("finish simulation!!!!!!!!!!!!!!!!!")

if __name__ == "__main__":

    start_time = time.time()
    print("start_time ", time.asctime(time.localtime(start_time)))

    action_ls = [
                    # 0, 
                    # 1, 
                    # 2, 
                    # 3, 
                    # 3.8, 
                    # 4, 
                    # 5, 
                    # 7,  
                    # 9, # "change_3_5", 
                    # 10, # "change_5_3",
                    # 11,
                    # 12,
                    # 13,
                    14
                 ]  
    with Pool(processes=6) as pool:
        pool.map(Simulate, action_ls)

    end_time = time.time()
    print("end_time ",time.asctime(time.localtime(end_time)))
    print("total elapsed time ",end_time-start_time)

    # Simulate(0)