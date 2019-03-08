import gym
from gym.utils import seeding
from tensorflow import keras
import numpy as np

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

    def __init__(self):
        self.state_idx = 0
        self.memory = np.zeros(shape=[EPISODE_MEMORY, NUM_FEATURES + 1])
        self.state_model = keras.models.load_model(STATE_MODEL_DESC)
        self.mortality_model = keras.models.load_model(MORTALITY_MODEL_DESC)
        self.starting_states = np.load(STARTING_STATES_VALUES)['sepsis_starting_states']
        self.seed()
        self.s = self.starting_states[np.random.randint(0, len(self.starting_states))]
        self.memory[-1 * (self.state_idx % EPISODE_MEMORY) - 1] = self.s
        return

    def step(self, action):
        next_state = self.state_model.predict(np.expand_dims(self.memory, 0))
        reward = self.mortality_model.predict(np.expand_dims(self.memory, 0))
        
        done = reward == [0, 0, 1]
        return next_state, reward, done, {"prob" : 1}

    def reset(self):
        self.memory = np.zeros(shape=[EPISODE_MEMORY, NUM_FEATURES])
        self.s = self.starting_states[np.random.randint(0, len(self.starting_states))]
        self.memory[-1 * (self.state_idx % EPISODE_MEMORY) - 1] = self.s
        return self.s

    def seed(self, seed=None):
        seed = seeding.np_random(seed)
        return [seed]
