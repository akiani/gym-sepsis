from gym_sepsis.envs.sepsis_env import SepsisEnv, SepsisEnvVariational

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


if __name__ == '__main__':
    unittest.main()