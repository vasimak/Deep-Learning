import gym
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import keras
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = gym.make("LunarLander-v2")
# env = gym.make("ALE/AirRaid-v5")
states = env.observation_space.shape[0]
actions = env.action_space.n


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(1, states)))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(actions, activation='linear'))



agent = DQNAgent(

    model=model,
    memory=SequentialMemory(limit=50000,window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions=actions,
    nb_steps_warmup=20,
    target_model_update=0.01
)


agent.compile(tf.keras.optimizers.legacy.Adam(learning_rate=10e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-7,name='Adam'),metrics=['mae'])


agent.fit(env,nb_steps=50000,visualize=False,verbose=1)

results = agent.test(env,nb_episodes=10,visualize=True)
print(np.mean(results.history["episode_reward"]))

env.close()
