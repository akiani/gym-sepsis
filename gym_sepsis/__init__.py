from gym.envs.registration import register

register(
    id='sepsis-v0',
    entry_point='gym_sepsis.envs:SepsisEnv',
)
