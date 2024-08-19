# from gymnasium.envs.registration import register


# register(
#      id="Custom_Envs/SubProcess_CommuteEnv_A_mu-v0",
#      entry_point="Custom_Envs.envs:SubProcess_CommuteEnv_A_mu",
#      max_episode_steps=30,
# )

from gymnasium.envs.registration import register

# Register your custom environment
register(
    id="SubProcess_CommuteEnv_A_mu",
    entry_point="gym_custom_env.envs:SubProcess_CommuteEnv_A_mu",
    max_episode_steps=60,
)

register(
    id="SubProcess_CommuteEnv_A_sigma",
    entry_point="gym_custom_env.envs:SubProcess_CommuteEnv_A_sigma",
    max_episode_steps=60,
)

register(
    id="SubProcess_CommuteEnv_mu_sigma",
    entry_point="gym_custom_env.envs:SubProcess_CommuteEnv_mu_sigma",
    max_episode_steps=60,
)

register(
    id="SubProcess_CommuteEnv_A_mu_sigma",
    entry_point="gym_custom_env.envs:SubProcess_CommuteEnv_A_mu_sigma",
    max_episode_steps=60,
)

# register(
#      id="Custom_Envs/CommuteEnv",
#      entry_point="Custom_Envs.envs:CommuteEnv_vanilla",
#      max_episode_steps=30,
# )

# register(
#      id="Custom_Envs/CommuteEnv-v1",
#      entry_point="Custom_Envs.envs:CommuteEnv_initialize_best_toll",
#      max_episode_steps=30,
# )

# register(
#      id="Custom_Envs/CommuteEnv-v2",
#      entry_point="Custom_Envs.envs:CommuteEnv_one_dim_action",
#      max_episode_steps=30,
# )


# register(
#      id="Custom_Envs/CommuteEnv-v3",
#      entry_point="Custom_Envs.envs:CommuteEnv_reward_shifting",
#      max_episode_steps=30,
# )

# register(
#      id="Custom_Envs/CommuteEnv-v4",
#      entry_point="Custom_Envs.envs:CommuteEnv",
# )

# register(
#      id="Custom_Envs/CommuteEnv-v5",
#      entry_point="Custom_Envs.envs:SubProcess_CommuteEnv_A", # optimize A, mu
# )

# register(
#      id="Custom_Envs/CommuteEnv-v6",
#      entry_point="Custom_Envs.envs:SubProcess_CommuteEnv_mu", # optimize mu
# )

# register(
#      id="Custom_Envs/CommuteEnv-v7",
#      entry_point="Custom_Envs.envs:SubProcess_CommuteEnv_sigma", # optimize mu
# )

# register(
#      id="Custom_Envs/CommuteEnv-v8",
#      entry_point="Custom_Envs.envs:SubProcess_CommuteEnv_A_mu", # optimize mu
# )

# register(
#      id="Custom_Envs/CommuteEnv-v9",
#      entry_point="Custom_Envs.envs:SubProcess_CommuteEnv_A_sigma", # optimize mu
# )

# register(
#      id="Custom_Envs/CommuteEnv-v10",
#      entry_point="Custom_Envs.envs:SubProcess_CommuteEnv_mu_sigma", # optimize mu
# )
# register(
#      id="Custom_Envs/CommuteEnv-v11",
#      entry_point="Custom_Envs.envs:SubProcess_CommuteEnv_A_mu_sigma", # optimize mu
# )