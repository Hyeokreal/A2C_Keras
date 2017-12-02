import gym
import time
import random
import threading
import numpy as np
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv2D
from keras import backend as K
from multi_env import SubprocVecEnv
from utils import make_atari, wrap_deepmind, discount_with_dones


class Runner(object):
    def __init__(self, env, action_size=4, nsteps=5, nstack=4, gamma=0.99, nh=84, nw=84, nc=1):
        self.env = env
        self.nh, self.nw, self.nc = nh, nw, nc
        nenv = env.num_envs
        self.action_size = action_size
        self.batch_ob_shape = (nenv * nsteps, nh, nw, nc * nstack)
        self.obs = np.zeros((nenv, nh, nw, nc * nstack), dtype=np.uint8)
        obs = env.reset()
        self.update_obs(obs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.dones = [False for _ in range(nenv)]
        self.state_size = (84, 84, 4)

        self.step_actor, self.step_critic = self.build_step_model()

    def build_step_model(self):
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

    def update_obs(self, obs):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce
        # IPC overhead
        self.obs = np.roll(self.obs, shift=-self.nc, axis=3)
        self.obs[:, :, :, -self.nc:] = obs.reshape(-1, self.nh, self.nw, self.nc)

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        for n in range(self.nsteps):

            ob = self.obs
            floatob = self.obs/255.
            print(type(ob))
            print(type(floatob))
            print(ob)
            print(floatob)
            policy = self.step_actor.predict(self.obs/ 255.)


            # policy = self.step_actor.predict(self.obs)
            # actions = np.argmax(policy, 1)


            actions = []
            for i in range(policy.shape[0]):
                action = np.random.choice(self.action_size, 1, p=policy[i])
                actions.append(action)
            actions = np.array(actions)


            values = self.step_critic.predict(self.obs)
            values = values[:, 0]

            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)

            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n] * 0
            self.update_obs(obs)

            mb_rewards.append(rewards)

        mb_dones.append(self.dones)
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)

        mb_dones = mb_dones[:, 1:]

        last_values = self.step_critic.predict(self.obs).tolist()

        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards + value, dones + [0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards

        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()

        return mb_obs, mb_actions, mb_rewards, mb_values

    def update_step_model(self, critic_weights, actor_weights):
        self.step_critic.set_weights(critic_weights)
        self.step_actor.set_weights(actor_weights)

if __name__ == "__main__":
    def make_env(rank):
        def _thunk():
            env = make_atari('BreakoutNoFrameskip-v4')
            env.seed(0 + rank)
            return wrap_deepmind(env)

        return _thunk


    env = SubprocVecEnv([make_env(i) for i in range(10)])

    runner = Runner(env)

    s, a, v, r = runner.run()

    print(s.shape)
    print(a.shape)
    print(v.shape)
    print(r.shape)
    print(a)
