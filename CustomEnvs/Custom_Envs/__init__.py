from gymnasium.envs.registration import register

register(
     id="Custom_Envs/CommuteEnv-v0",
     entry_point="Custom_Envs.envs:CommuteEnv",
     max_episode_steps=30,
)