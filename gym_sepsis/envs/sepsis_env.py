import gym
from gym.utils import seeding
from tensorflow import keras
import numpy as np
import os
from collections import deque
import pandas as pd
from gym import spaces

STATE_MODEL_DESC = "model/sepsis_states.model"
MORTALITY_MODEL_DESC = "model/sepsis_mortality.model"
STARTING_STATES_VALUES = "model/sepsis_starting_states.npz"

NUM_FEATURES = 46
NUM_ACTIONS = 24

EPISODE_MEMORY = 10


class ObservationSpace(object):
    def __init__(self, shape):
        self.shape = shape


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
        self.state_model = keras.models.load_model(
            os.path.join(module_path, STATE_MODEL_DESC))
        self.mortality_model = keras.models.load_model(
            os.path.join(module_path, MORTALITY_MODEL_DESC))
        self.starting_states = np.load(
            os.path.join(module_path,
                         STARTING_STATES_VALUES))['sepsis_starting_states']
        self.seed()
        self.action_space = spaces.Discrete(24)

        # use a pixel to represent next state
        self.observation_space = ObservationSpace([1, 1, 46])
        self.reset(starting_state=starting_state)
        return

    def step(self, action):
        # create memory of present
        self.memory.append(np.append(self.s.reshape((1, 46)), action))
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
        self.s = next_state.reshape(1, 1, 46)
        self.state_idx += 1
        self.rewards.append(reward)
        self.dones.append(done)
        return self.s, reward, done, {"prob": 1}

    def reset(self, starting_state=None):
        self.rewards = []
        self.dones = []
        self.state_idx = 0
        self.memory = deque(
            [np.zeros(shape=[NUM_FEATURES + 1])] * 10, maxlen=10)
        if starting_state is None:
            self.s = self.starting_states[np.random.randint(
                0, len(self.starting_states))][:-1]
        else:
            self.s = starting_state

        self.s = self.s.reshape(1, 1, 46)

        if self.verbose:
            print("starting state:", self.s)
        return self.s

    def seed(self, seed=None):
        seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='ansi'):
        columns = [
            'ALBUMIN', 'ANION GAP', 'BANDS', 'BICARBONATE', 'BILIRUBIN', 'BUN',
            'CHLORIDE', 'CREATININE', 'DiasBP', 'Glucose', 'GLUCOSE',
            'HeartRate', 'HEMATOCRIT', 'HEMOGLOBIN', 'INR', 'LACTATE',
            'MeanBP', 'PaCO2', 'PLATELET', 'POTASSIUM', 'PT', 'PTT',
            'RespRate', 'SODIUM', 'SpO2', 'SysBP', 'TempC', 'WBC', 'age',
            'is_male', 'race_white', 'race_black', 'race_hispanic',
            'race_other', 'height', 'weight', 'vent', 'sofa', 'lods', 'sirs',
            'qsofa', 'qsofa_sysbp_score', 'qsofa_gcs_score',
            'qsofa_resprate_score', 'elixhauser_hospital',
            'blood_culture_positive', 'action'
        ]
        df = pd.DataFrame(self.memory, columns=columns, index=range(0, 10))
        print(df)
