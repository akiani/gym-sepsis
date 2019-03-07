import gym
from gym.utils import seeding
from tensorflow import keras
import numpy as np

STATE_MODEL_DESC = "model/sepsis_states.json"
STATE_MODEL_WEIGHTS = "model/sepsis_states.h5py"

MORTALITY_MODEL_WEIGHTS = "model/sepsis_mortality.h5py"
MORTALITY_MODEL_DESC = "model/sepsis_mortality.json"

STARTING_STATES_VALUES = "model/sepsis_starting_stats.npz"

NUM_FEATURES = 46

EPISODE_MEMORY = 10

def load_model(model_path, weights_path):
    with open(model_path, 'r') as json_file:
        print("Loading model from %s with weights from %s..." % (model_path, weights_path))
        loaded_model_json = json_file.read()
        loaded_model = keras.model_from_json(loaded_model_json)
        loaded_model.load_weights(weights_path)
    return loaded_model


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
        self.memory = np.zeros(shape=[EPISODE_MEMORY, NUM_FEATURES])
        self.state_model = load_model(STATE_MODEL_DESC, STATE_MODEL_WEIGHTS)
        self.mortality_model = load_model(MORTALITY_MODEL_DESC, MORTALITY_MODEL_WEIGHTS)
        self.starting_states = np.loadz(STARTING_STATES_VALUES)['start_states']
        self.seed()
        self.s = self.starting_states[np.random.randint(0, len(self.starting_states))]
        self.memory[-1 * (self.state_idx % EPISODE_MEMORY) - 1] = self.s
        return

    def step(self, action):
        next_state = self.state_model.predict(self.memory)
        reward = self.mortality_model.predict(self.memory)
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
