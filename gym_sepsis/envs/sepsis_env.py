import gym
from gym.utils import seeding
from tensorflow import keras
import numpy as np
import os
from collections import deque
import pandas as pd
from gym import spaces
import tensorflow as tf

# no vae
STATE_MODEL = "model/sepsis_states.model"
TERMINATION_MODEL = "model/sepsis_termination.model"
OUTCOME_MODEL = "model/sepsis_outcome.model"
STARTING_STATES_VALUES = "model/sepsis_starting_states.npz"

# with vae
STATE_MODEL_VAE = "model/sepsis_states_vae.model"
TERMINATION_MODEL_VAE = "model/sepsis_termination_vae.model"
OUTCOME_MODEL_VAE = "model/sepsis_outcome_vae.model"
VAE_MODEL = "model/vae_noklloss.model"

# with small vae
STATE_MODEL_VAE_SMALL = "model/sepsis_states_vae_smaller.model"
TERMINATION_MODEL_VAE_SMALL = "model/sepsis_termination_vae_small.model"
OUTCOME_MODEL_VAE_SMALL = "model/sepsis_outcome_vae_small.model"
VAE_MODEL_SMALL = "model/vae_noklloss_small.model"

NUM_FEATURES_VAE = 32 # 46 + action + state index
NUM_FEATURES = 48  # 46 + action + state index
NUM_ACTIONS = 24

EPISODE_MEMORY = 10

features = ['ALBUMIN', 'ANION GAP', 'BANDS', 'BICARBONATE',
            'BILIRUBIN', 'BUN', 'CHLORIDE', 'CREATININE', 'DiasBP', 'Glucose',
            'GLUCOSE', 'HeartRate', 'HEMATOCRIT', 'HEMOGLOBIN', 'INR', 'LACTATE',
            'MeanBP', 'PaCO2', 'PLATELET', 'POTASSIUM', 'PT', 'PTT', 'RespRate',
            'SODIUM', 'SpO2', 'SysBP', 'TempC', 'WBC', 'age', 'is_male',
            'race_white', 'race_black', 'race_hispanic', 'race_other', 'height',
            'weight', 'vent', 'sofa', 'lods', 'sirs', 'qsofa', 'qsofa_sysbp_score',
            'qsofa_gcs_score', 'qsofa_resprate_score', 'elixhauser_hospital',
            'blood_culture_positive', 'action', 'state_idx']


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
        self.state_model = keras.models.load_model(os.path.join(module_path, STATE_MODEL))
        self.termination_model = keras.models.load_model(os.path.join(module_path, TERMINATION_MODEL))
        self.outcome_model = keras.models.load_model(os.path.join(module_path, OUTCOME_MODEL))
        self.starting_states = np.load(os.path.join(module_path, STARTING_STATES_VALUES))['sepsis_starting_states']
        self.seed()
        self.action_space = spaces.Discrete(24)

        # use a pixel to represent next state
        self.observation_space = spaces.Box(low=0, high=NUM_ACTIONS, shape=(NUM_FEATURES-2, 1, 1),
                                            dtype=np.float32)
        self.reset(starting_state=starting_state)
        return


    def step(self, action):
        # create memory of present
        self.memory.append(np.append(np.append(self.s.reshape((1, NUM_FEATURES - 2)), action), self.state_idx))
        if self.verbose:
            print("running on memory: ", self.memory)

        memory_array = np.expand_dims(self.memory, 0)
        next_state = self.state_model.predict(memory_array[:, :, :-1])

        # overwrite constant variables (these should't change during episode)
        constants = ['age', 'race_white', 'race_black', 'race_hispanic',
                     'race_other', 'height', 'weight']
        for constant in constants:
            idx = features.index(constant)
            val = self.state_0[idx]
            next_state[0, idx] = val

        termination = self.termination_model.predict(memory_array)
        outcome = self.outcome_model.predict(memory_array)

        termination_categories = ['continue', 'done']
        outcome_categories = ['death', 'release']

        termination_state = termination_categories[np.argmax(termination)]
        outcome_state = outcome_categories[np.argmax(outcome)]

        reward = 0
        done = False

        if termination_state == 'done':
            done = True
            if outcome_state == 'death':
                reward = -15
            else:
                reward = 15

        # keep next state in memory
        self.s = next_state.reshape(46, 1, 1)
        self.state_idx += 1
        self.rewards.append(reward)
        self.dones.append(done)
        return self.s, reward, done, {"prob" : 1}

    def reset(self, starting_state=None):
        self.rewards = []
        self.dones = []
        self.state_idx = 0
        self.memory = deque([np.zeros(shape=[NUM_FEATURES])] * 10, maxlen=10)
        if starting_state is None:
            self.s = self.starting_states[np.random.randint(0, len(self.starting_states))][:-1]
        else:
            self.s = starting_state

        self.s = self.s.reshape(NUM_FEATURES - 2, 1, 1)
        self.state_0 = np.copy(self.s)

        if self.verbose:
            print("starting state:", self.s)
        return self.s

    def seed(self, seed=None):
        seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='ansi'):
        df = pd.DataFrame(self.memory, columns=features, index=range(0, 10))
        print(df)


class SepsisEnvVariational(gym.Env):
    """
    Built from trained models on top of the MIMIC dataset, this
    Environment simulates the behavior of the Sepsis patient
    in response to medical interventions and uses a VAE to reduce
    dimensionality.

    For details see: https://github.com/chriswang1999/Sepsis 
    """
    metadata = {'render.modes': ['ansi']}

    def __init__(self, starting_state=None, verbose=False):
        module_path = os.path.dirname(__file__)
        self.verbose = verbose
        self.vae_model = keras.models.load_model(os.path.join(module_path, VAE_MODEL), compile=False)
        self.encoder_model = self.get_encoder(self.vae_model)
        self.decoder_model = self.get_decoder(self.vae_model)
        self.state_model = keras.models.load_model(os.path.join(module_path, STATE_MODEL_VAE))
        self.termination_model = keras.models.load_model(os.path.join(module_path, TERMINATION_MODEL_VAE))
        self.outcome_model = keras.models.load_model(os.path.join(module_path, OUTCOME_MODEL_VAE))
        self.starting_states = np.load(os.path.join(module_path, STARTING_STATES_VALUES))['sepsis_starting_states']
        self.seed()
        self.action_space = spaces.Discrete(24)
        self.observation_space = spaces.Box(low=0, high=NUM_ACTIONS, shape=(NUM_FEATURES - 2, 1, 1),
                                            dtype=np.float32)
        self.reset(starting_state=starting_state)
        return

    def get_decoder(self, vae_model):
        input_layer = keras.layers.Input(shape=(30,))
        decoder_layer1 = vae_model.layers[7]
        decoder_layer2 = vae_model.layers[8]
        decoder_layer3 = vae_model.layers[9]
        decoder = tf.keras.Model(input_layer, decoder_layer3(decoder_layer2(decoder_layer1(input_layer))))
        return decoder

    def get_encoder(self, vae_model):
        input_img = tf.keras.layers.Input(shape=(46,))
        encoder_layer1 = vae_model.layers[0]
        encoder_layer2 = vae_model.layers[1]
        encoder_layer3 = vae_model.layers[2]
        encoder_layer4 = vae_model.layers[3]
        encoder_layer5 = vae_model.layers[4]
        encoder_layer6 = vae_model.layers[5]
        encoder_layer7 = vae_model.layers[6]
        intermediate = encoder_layer4(encoder_layer3(encoder_layer2(encoder_layer1(input_img))))
        output = encoder_layer7([encoder_layer5(intermediate), encoder_layer6(intermediate)])
        encoder = tf.keras.Model(input_img, output)
        return encoder

    def encode_state(self, state):
        return self.encoder_model.predict(state)

    def decode_state(self, encoded_state):
        return self.decoder_model.predict(encoded_state)

    def step(self, action):
        # create memory of present
        self.memory.append(np.append(np.append(self.s.reshape((1, NUM_FEATURES_VAE - 2)), action), self.state_idx))
        if self.verbose:
            print("running on memory: ", self.memory)

        memory_array = np.expand_dims(self.memory, 0)
        next_state = self.state_model.predict(memory_array[:, :, :-1])
        termination = self.termination_model.predict(memory_array)
        outcome = self.outcome_model.predict(memory_array)

        termination_categories = ['continue', 'done']
        outcome_categories = ['death', 'release']

        termination_state = termination_categories[np.argmax(termination)]
        outcome_state = outcome_categories[np.argmax(outcome)]

        reward = 0
        done = False

        if termination_state == 'done':
            done = True
            if outcome_state == 'death':
                reward = -15
            else:
                reward = 15

        # keep next state in memory
        self.s = next_state.reshape(30, 1, 1)
        self.state_idx += 1
        self.rewards.append(reward)
        self.dones.append(done)
        decoded_state = self.decode_state(self.s.reshape(1, 30))
        return decoded_state, reward, done, {"prob" : 1}

    def reset(self, starting_state=None):
        self.rewards = []
        self.dones = []
        self.state_idx = 0
        self.memory = deque([np.zeros(shape=[NUM_FEATURES_VAE])] * 10, maxlen=10)
        if starting_state is None:
            self.s = self.encode_state(self.starting_states[np.random.randint(0, len(self.starting_states))][:-1].reshape(1, 46))
        else:
            self.s = self.encode_state(starting_state.reshape(1, 46))

        self.s = self.s.reshape(NUM_FEATURES_VAE - 2, 1, 1)
        self.state_0 = np.copy(self.s)

        if self.verbose:
            print("starting state:", self.s)
        decoded_state = self.decode_state(self.s.reshape(1, 30))
        return decoded_state

    def render(self, mode='ansi'):
        df = pd.DataFrame(self.memory, index=range(0, 10))
        print(df)

    def seed(self, seed=None):
        seed = seeding.np_random(seed)
        return [seed]


class SepsisEnvVariationalSmall(SepsisEnvVariational):
    def __init__(self, starting_state=None, verbose=False):
        module_path = os.path.dirname(__file__)
        self.verbose = verbose
        self.vae_model = keras.models.load_model(os.path.join(module_path, VAE_MODEL_SMALL), compile=False)
        self.encoder_model = self.get_encoder(self.vae_model)
        self.decoder_model = self.get_decoder(self.vae_model)
        self.state_model = keras.models.load_model(os.path.join(module_path, STATE_MODEL_VAE_SMALL))
        self.termination_model = keras.models.load_model(os.path.join(module_path, TERMINATION_MODEL_VAE_SMALL))
        self.outcome_model = keras.models.load_model(os.path.join(module_path, OUTCOME_MODEL_VAE_SMALL))
        self.starting_states = np.load(os.path.join(module_path, STARTING_STATES_VALUES))['sepsis_starting_states']
        self.seed()
        self.action_space = spaces.Discrete(24)
        self.observation_space = spaces.Box(low=0, high=NUM_ACTIONS, shape=(NUM_FEATURES - 2, 1, 1),
                                            dtype=np.float32)
        self.reset(starting_state=starting_state)
        return
    
    
