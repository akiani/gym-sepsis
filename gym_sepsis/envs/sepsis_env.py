import gym
from gym.utils import seeding
from tensorflow import keras
import numpy as np
import os
from collections import deque 


STATE_MODEL_DESC = "model/sepsis_states.model"
MORTALITY_MODEL_DESC = "model/sepsis_mortality.model"
STARTING_STATES_VALUES = "model/sepsis_starting_states.npz"

NUM_FEATURES = 46
NUM_ACTIONS = 24

EPISODE_MEMORY = 10


class SepsisEnv(gym.Env):
    """
    Built from trained models on top of the MIMIC dataset, this
    Environment simulates the behavior of the Sepsis patient
    in response to medical interventions.
    For details see: https://github.com/akiani/gym-sepsis 
    """
    metadata = {'render.modes': ['ansi']}

    def __init__(self, starting_state=None, verbose=False):
        module_path = os.path.dirname(__file__)
        self.verbose = verbose
        self.state_model = keras.models.load_model(os.path.join(module_path, STATE_MODEL_DESC))
        self.mortality_model = keras.models.load_model(os.path.join(module_path, MORTALITY_MODEL_DESC))
        self.starting_states = np.load(os.path.join(module_path, STARTING_STATES_VALUES))['sepsis_starting_states']
        self.seed()
        self.reset(starting_state=starting_state)
        return

    def step(self, action):
        # create memory of present
        self.memory.append(np.append(self.s, action))
        if self.verbose:
            print("running on memory: ", self.memory)

        next_state = self.state_model.predict(np.expand_dims(self.memory, 0))
        reward = self.mortality_model.predict(np.expand_dims(self.memory, 0))

        reward_categories = ['continue', 'died', 'released']
        reward_state = reward_categories[np.argmax(reward)]

        reward = 0
        done = False

        if reward_state == 'died':
            reward = -15
            done = True
        elif reward_state == 'released':
            reward = 15
            done = True

        # keep next state in memory
        self.s = next_state
        self.state_idx += 1
        self.rewards.append(reward)
        self.dones.append(done)
        return next_state, reward, done, {"prob" : 1}

    def reset(self, starting_state=None):
        self.rewards = []
        self.dones = []
        self.state_idx = 0
        self.memory = deque([np.zeros(shape=[NUM_FEATURES + 1])] * 10, maxlen=10)
        if starting_state is None:
            self.s = self.starting_states[np.random.randint(0, len(self.starting_states))][:-1]
        else:
            self.s = starting_state

        if self.verbose:
            print("starting state:", self.s)
        return self.s

    def seed(self, seed=None):
        seed = seeding.np_random(seed)
        return [seed]