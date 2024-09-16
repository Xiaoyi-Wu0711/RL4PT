This is a customed environment based on Gymnasium, which is composed of macroscopic fundamental diagram(MFD) and tradable credit systems(TCS). 

To install the package in local computer: 
`pip install -e CustomEnvs`
`pip install -r requirements.txt`

To use the package:
`import Custom_Envs`

Reference: 

1. TCS: Chen, Siyu, et al. "Market design for tradable mobility credits." Transportation Research Part C: Emerging Technologies 151 (2023): 104121.

2. MFD: Liu, Renming, et al. "Managing network congestion with a trip-and area-based tradable credit scheme." Transportmetrica B: Transport Dynamics 11.1 (2023): 434-462.

3. Gymnasium <a href="https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/">tutorial</a>.

4. run: 
python3 main_hyperparam.py --experiment_name "check optimization function setup" --one_dim True --best_toll_initialization False --reward_shifting True --absolute_change_mode False --n_trial 100 --train_episode 20 --simulation_day_num 30 --evaluation_time_episode 20 --model_param "ppo"

5. Run with slurm:
SBATCH main.job
