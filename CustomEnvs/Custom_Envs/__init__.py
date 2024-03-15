from gymnasium.envs.registration import register

register(
     id="Custom_Envs/CommuteEnv-v0",
     entry_point="Custom_Envs.envs:CommuteEnv_vanilla",
     max_episode_steps=30,
)

register(
     id="Custom_Envs/CommuteEnv-v1",
     entry_point="Custom_Envs.envs:CommuteEnv_initialize_best_toll",
     max_episode_steps=30,
)

register(
     id="Custom_Envs/CommuteEnv-v2",
     entry_point="Custom_Envs.envs:CommuteEnv_one_dim_action",
     max_episode_steps=30,
)


register(
     id="Custom_Envs/CommuteEnv-v3",
     entry_point="Custom_Envs.envs:CommuteEnv_reward_shifting",
     max_episode_steps=30,
)

register(
     id="Custom_Envs/CommuteEnv-v4",
     entry_point="Custom_Envs.envs:CommuteEnv",
)

register(
     id="Custom_Envs/CommuteEnv-v5",
     entry_point="Custom_Envs.envs:SubProcess_CommuteEnv",
)
