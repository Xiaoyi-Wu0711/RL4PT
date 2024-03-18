import os
import warnings
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
import os
import warnings
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from typing import Any, Dict, Type, Union, List, Optional, Callable, Tuple
import matplotlib.pyplot as plt
from stable_baselines3.common.logger import HParam
import numpy as np


class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_path: str, verbose=1):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_path = save_path
        
    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        save_path_name = os.path.join(self.save_path, "vecnormalize.pkl")
        self.model.get_vec_normalize_env().save(save_path_name)
        print("Saved vectorized and normalized environment to {}".format(save_path_name))


def lrsched():
  def reallr(progress):
    lr = 0.001
    if progress < 0.2: # pro
      lr = 0.0001
    return lr
  return reallr



class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
    
    # def _on_training_start(self) -> None:
    #    pass
        # print("self.model.get_parameters() ", self.model.get_parameters())

        # hparam_dict = {
        #     "algorithm": self.model.__class__.__name__,
        #     "learning rate": self.model.learning_rate,
        #     "gamma": self.model.gamma,
        #     "gae_lambda": self.model.gae_lambda,
        #     "n_steps": self.model.n_steps,
        #     "num_timesteps": self.model.num_timesteps,
        #     "action_space": self.model.action_space,
        #     "observation_space": self.model.observation_space,

        # }
        
        # metric_dict: Dict[str, Union[float, int]] = {
        #     "eval/mean_reward": 0,
        #     "rollout/ep_rew_mean": 0,
        #     "rollout/ep_len_mean": 0,
        #     "train/value_loss": 0,
        #     "train/explained_variance": 0,
        # }
        # self.logger.record(
        #     "hparams",
        #     HParam(hparam_dict, metric_dict),
        #     exclude=("stdout", "log", "json", "csv"),
        # )
    
    def _on_rollout_end(self,) -> None:
        num_envs = self.training_env.num_envs

        # Check if the episode is done
        
        self.logger.record("rollout/mean_value", np.mean(self.locals["rollout_buffer"].values))
        self.logger.record("rollout/mean_rew", np.mean(self.locals["rollout_buffer"].rewards))

        return
            
    def _on_step(self) -> bool:
       return True
    #     # print(self.locals)
    #     # print(" ")
    #     # print(self.globals)
    #     print(" --------- ")

    #     # Log scalar value (here a random variable)
    #     sw = self.locals["infos"][0]["sw"]
    #     pt_share_number = self.locals["infos"][0]["pt_share_number"]
    #     market_price = self.locals["infos"][0]["market_price"]
    #     AITT_daily = self.locals["infos"][0]["AITT_daily"]
    #     figure = plt.figure()
    #     self.logger.record("trajectory/sw", sw,)
    #     self.logger.record("trajectory/pt_share_number", pt_share_number, )
    #     self.logger.record("trajectory/market_price", market_price)
    #     self.logger.record("trajectory/AITT_daily", AITT_daily)
    #     self.logger.dump(self.num_timesteps)

    #     return True