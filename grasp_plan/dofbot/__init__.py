from gymnasium.envs.registration import register

# Register table environment
register(
    id='TableSceneEnv-v0',
    entry_point='dofbot.envs.table_scene_env:TableSceneEnv',
    max_episode_steps=500,
    reward_threshold=100.0,  # Optional
)