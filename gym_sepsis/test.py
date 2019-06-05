from gym_sepsis.envs.sepsis_env import SepsisEnv, SepsisEnvVariational
from gym_sepsis.envs.sepsis_env_bayesian import SepsisEnvBayesian

import unittest


class TestEnvironment(unittest.TestCase):
    def test_init_normal(self):
        env = SepsisEnv()
        env.step(1)
        env.render()

    def test_init_vae(self):
        env = SepsisEnvVariational()
        env.step(1)
        env.render()
        env.step(1)
        env.render()

    def test_init_bayesian(self):
        env = SepsisEnvBayesian()
        env.step(1)
        env.render()
        env.step(1)
        env.render()


if __name__ == '__main__':
    unittest.main()