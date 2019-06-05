from gym.envs.registration import register

register(
    id='sepsis-v0',
    entry_point='gym_sepsis.envs:SepsisEnv',
)

register(
    id='sepsis-vae-v0',
    entry_point='gym_sepsis.envs:SepsisEnvVariational',
)

register(
    id='sepsis-vae-small-v0',
    entry_point='gym_sepsis.envs:SepsisEnvVariationalSmall',
)

register(
    id='sepsis-bayesian-v0',
    entry_point='gym_sepsis.envs:SepsisEnvBayesian',
)

