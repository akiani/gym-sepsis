from gym_sepsis.envs.sepsis_env import SepsisEnv

try:
    env = SepsisEnv()
    env.step(1)
    env.render()
    print("Test passed.")
except Exception as ex:
    raise ex
