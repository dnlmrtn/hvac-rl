'''
Training roadmap:
- Get weather as training data
- Use simulated temperature dynamics to create env:
    Training tuple: (indoor_temp1, outdoor_temp1, time_1, action, indor_temp2)
'''


import tensorflow as tf
import numpy as np
import random
from gym import Env
from gym.spaces import Discrete, Box
import pandas as pd
from datetime import datetime, timedelta
# -------------------------
# Weather and Zone Simulation
# -------------------------
# Temp dynamics
# delta_indoor_temp = heat transfer from outside + heat transfer from fresh air damper + heat transfer from hvac
# delta_indoor_temp = 0.1(outdoor_temp - indoor_temp) + 0.1*damper*(outdoor_temp-indoor_temp) + hvac

weather_df = pd.read_csv('./data/weather_data.csv')
temps = weather_df['temperature']

SET_TEMP = 22.4
time_format = "%H:%M:%S"
start_time = datetime.strptime("16:00:00", time_format)


class hvacEnv(Env):
    def __init__(self):
        # Initialize actions, observations
        self.action_space = Discrete(10)

        low = np.array([-10, -30], dtype=np.float32)
        high = np.array([50, 50], dtype=np.float32)
        self.observation_space = Box(low, high, dtype=np.float32)
        self.idx = weather_df.shape[0]
        self.state = (SET_TEMP, temps.values[-1])

        pass

    def step(self, action):
        if self.idx == 0:
            done = True
        else:
            self.idx -= 1
            done = False

        indoor_temp_1 = self.state[0]
        outdoor_temp_1 = self.state[1]

        # Calculate new indoor temp
        new_indoor_temp = 0.1(outdoor_temp_1 - indoor_temp_1) + \
            0.1*action[1]*(outdoor_temp_1-indoor_temp_1) + action[0]

        # Update to next outdoor temp
        new_outdoor_temp = temps[self.idx]

        if 22 < new_indoor_temp < 23:
            reward = 1
        else:
            reward = -1

        # Assign placeholder for info
        info = {}
        # Assign new state
        self.state = (new_indoor_temp, new_outdoor_temp)
        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):
        self.idx = weather_df.shape[0]
        self.state = (SET_TEMP, temps.values[-1])
        return self.state


test = hvacEnv()
# -------------------------
# DQN Agent
# -------------------------
# 10 Actions:
# Damper Position 15%
# Phase 1 cool - 1
# Phase 2 cool - 2
# Phase 1 heat + 1
# Phase 2 heat + 2
# Fan + 0
# Damper position 100%
# Phase 1 cool - 1
# Phase 2 cool - 2
# Phase 1 heat + 1
# Phase 2 heat + 2
# Fan + 0


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(
            24, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                    np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.train_on_batch(state, target_f)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# -------------------------
# Training Loop
# -------------------------
TARGET_TEMP = 22.4  # Target indoor temperature in °C
