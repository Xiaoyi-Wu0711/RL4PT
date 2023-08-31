from gymnasium.envs.registration import register

register(
     id="examples/CommuteEnv-v0",
     entry_point="examples.envs:CommuteEnv",
     max_episode_steps=3000,
)