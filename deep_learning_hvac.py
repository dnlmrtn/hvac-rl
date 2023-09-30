'''
Training roadmap:
- Get weather as training data
- Use simulated temperature dynamics to create env:
    Training tuple: (indoor_temp1, outdoor_temp1, time) -> (Actions)
'''


import tensorflow as tf
import numpy as np
import random
import gym
# -------------------------
# Weather and Zone Simulation
# -------------------------
# Temp dynamics
# delta_indoor_temp = heat transfer from outside + heat transfer from fresh air damper + heat transfer from hvac
# delta_indoor_temp = 0.1(outdoor_temp - indoor_temp) + 0.1*damper*(outdoor_temp-indoor_temp) + hvac


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
