import gym
import os
import random
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv2D
from keras import backend as K
from keras.utils.np_utils import to_categorical
from skimage.color import rgb2gray
from skimage.transform import resize
from multi_env import SubprocVecEnv
from utils import make_atari, wrap_deepmind, discount_with_dones

# global variables for A3C
MAX_STEP = 80000000

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
        self.critic_lr = 0.0002

        # create model for actor and critic network
        self.actor, self.critic = self.build_model()

        # method for training actor and critic network
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

    # approximate policy and value using Neural Network
    # actor -> state is input and probability of each action is output of network
    # critic -> state is input and value of state is output of network
    # actor and critic network share first hidden layer
    def build_model(self):
        input = Input(shape=self.state_size)
        conv = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input)
        conv = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv)
        conv = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv)
        conv = Flatten()(conv)
        fc = Dense(512, activation='relu')(conv)
        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1, activation='linear')(fc)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()

        return actor, critic

    def train_model(self, s, a, r, v):
        a = to_categorical(a, self.action_size)
        s /= 255.

        self.optimizer[0]([s, a, r - v])
        self.optimizer[1]([s, r])

    def get_action(self, obs):
        obs = np.float32(obs / 255.)
        policy = self.actor.predict(obs)[0]
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index

    # make loss function for Policy Gradient
    # [log(action probability) * advantages] will be input for the back prop
    # we add entropy of action probability to loss
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])

        policy = self.actor.output

        good_prob = K.sum(action * policy, axis=1)
        eligibility = K.log(good_prob + 1e-10) * advantages
        actor_loss = -K.sum(eligibility)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)

        loss = actor_loss + 0.01 * entropy
        optimizer = RMSprop(lr=self.actor_lr, rho=0.99, epsilon=0.01)
        # optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantages], [loss], updates=updates)

        return train

    # make loss function for Value approximation
    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None,))

        value = self.critic.output

        loss = K.mean(K.square(discounted_reward - value))

        optimizer = RMSprop(lr=self.critic_lr, rho=0.99, epsilon=0.01)
        # optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, discounted_reward], [loss], updates=updates)
        return train

    def load_model(self, name):
        self.actor.load_weights(os.path.join(model_path, name + "_actor.h5"))
        self.critic.load_weights(os.path.join(model_path, name + "_critic.h5"))

    def save_model(self, name):
        self.actor.save_weights(os.path.join(model_path, name + "_actor.h5"))
        self.critic.save_weights(os.path.join(model_path, name + "_critic.h5"))


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
        conv = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input)
        conv = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv)
        conv = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv)
        conv = Flatten()(conv)
        fc = Dense(512, activation='relu')(conv)
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

            obs = np.float32(self.obs / 255.)

            policy = self.step_actor.predict(obs)

            actions = []
            for i in range(policy.shape[0]):
                action = np.random.choice(self.action_size, 1, p=policy[i])
                actions.append(action)
            actions = np.array(actions)

            values = self.step_critic.predict(obs)
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

    def update_step_model(self, actor_weights, critic_weights):
        self.step_actor.set_weights(actor_weights)
        self.step_critic.set_weights(critic_weights)


def make_env(rank):
    def _thunk():
        env = make_atari('BreakoutNoFrameskip-v4')
        env.seed(0 + rank)
        return wrap_deepmind(env)

    return _thunk


def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":
    num_proc = 8
    env = SubprocVecEnv([make_env(i) for i in range(num_proc)])
    test_env = gym.make('BreakoutDeterministic-v4')
    # test_env = SubprocVecEnv([make_env(i) for i in range(1)])
    runner = Runner(env)
    agent = A2CAgent(action_size=4)

    for i in range(MAX_STEP//num_proc):
        s, a, v, r = runner.run()
        agent.train_model(s, a, r, v)
        runner.update_step_model(agent.actor.get_weights(), agent.critic.get_weights())

        if i % 1000 == 0:
            model_name = str(i) + 'th_model'
            agent.save_model(model_name)

            scores = []
            for e in range(5):
                score = 0
                stack_obs = np.zeros((1, 84, 84, 4), dtype=np.uint8)

                obs = test_env.reset()

                for _ in range(random.randint(1, 30)):
                    observe, _, _, _ = test_env.step(1)
                    observe = pre_processing(observe)
                    stack_obs = np.roll(stack_obs, shift=-1, axis=3)
                    stack_obs[:, :, :, -1:] = observe.reshape(-1, 84, 84, 1)

                while True:
                    action = agent.get_action(stack_obs / 255.)
                    next_obs, reward, done, _ = test_env.step(action)
                    next_obs = pre_processing(next_obs)
                    stack_obs = np.roll(stack_obs, shift=-1, axis=3)
                    stack_obs[:, :, :, -1:] = next_obs.reshape(-1, 84, 84, 1)

                    score += reward
                    obs = next_obs

                    if done:
                        scores.append(score)
                        break
            print("%d th iteration, mean score : %f " % (i, np.mean(scores)))
