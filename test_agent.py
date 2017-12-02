import gym
import os
import random
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv2D
from keras import backend as K
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.utils.np_utils import to_categorical

model_path = os.path.join(os.getcwd(), 'save_model')

if not os.path.isdir(model_path):
    os.mkdir(model_path)

class A2CAgent:
    def __init__(self, action_size):
        # environment settings
        self.state_size = (84, 84, 4)
        self.action_size = action_size

        self.discount_factor = 0.99
        self.no_op_steps = 30

        # optimizer parameters
        self.actor_lr = 0.0001
        self.critic_lr = 0.0001

        # create model for actor and critic network
        self.actor, self.critic = self.build_model()

    # approximate policy and value using Neural Network
    # actor -> state is input and probability of each action is output of network
    # critic -> state is input and value of state is output of network
    # actor and critic network share first hidden layer
    def build_model(self):
        input = Input(shape=self.state_size)
        conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(input)
        conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
        conv = Flatten()(conv)
        fc = Dense(256, activation='relu')(conv)
        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1, activation='linear')(fc)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()

        return actor, critic

    def get_action(self, obs):
        obs = np.float32(obs / 255.)
        policy = self.actor.predict(obs)[0]
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index

    def load_model(self, name):
        self.actor.load_weights(os.path.join(model_path, name + "_actor.h5"))
        self.critic.load_weights(os.path.join(model_path, name + "_critic.h5"))

    def save_model(self, name):
        self.actor.save_weights(os.path.join(model_path, name + "_actor.h5"))
        self.critic.save_weights(os.path.join(model_path, name + "_critic.h5"))

if __name__ == "__main__":
    test_env = gym.make('BreakoutNoFrameskip-v4')
    agent = A2CAgent(action_size=4)
    agent.load_model('model')


    def pre_processing(observe):
        processed_observe = np.uint8(
            resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
        return processed_observe

    scores = []
    for e in range(5):
        score = 0
        stack_obs = np.zeros((1, 84, 84, 4), dtype=np.uint8)

        obs = test_env.reset()

        for _ in range(4):
            observe, _, _, _ = test_env.step(1)
            observe = pre_processing(observe)
            stack_obs = np.roll(stack_obs, shift=-1, axis=3)
            stack_obs[:, :, :, -1:] = observe.reshape(-1, 84, 84, 1)

        while True:
            test_env.render()
            action = agent.get_action(stack_obs / 255.)
            next_obs, reward, done, _ = test_env.step(action)
            next_obs = pre_processing(next_obs)
            stack_obs = np.roll(stack_obs, shift=-1, axis=3)
            stack_obs[:, :, :, -1:] = next_obs.reshape(-1, 84, 84, 1)

            score += reward
            obs = next_obs

            if done:
                print("score : ", score)
                break
