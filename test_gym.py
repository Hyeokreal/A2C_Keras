import gym
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray

envids = [spec.id for spec in gym.envs.registry.all()]
print(len(envids)," 개의 환경이 존재합니다.")
for envid in sorted(envids):
    print(envid)

# game = random.choice(envids)
# print("picked game : ", game)
# env = gym.make(game)
# env = gym.make('BreakoutDeterministic-v4')
env = gym.make('BreakoutNoFrameskip-v4')

# 행동, 상태, 보상 확인

print(env.action_space)
print(env.observation_space)
print(env.reward_range)

# 행동의 개수 확인
print(env.action_space.n)

# 학습속도를 높이기 위해 흑백화면으로 전처리
def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe


# action test

for e in range(env.action_space.n):
    observation = env.reset()
    print("test for action : ", e)

    for _ in range(500):
        env.render()
        action = e
        env.step(action)


for i_episode in range(2):
    observation = env.reset()
    rewards = []
    while True:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if reward not in rewards:
            rewards.append(reward)
            # print("observation : ", observation[50])
            print("grey : ", pre_processing(observation)[50])
            print("reward : ", reward)
            print("done : ", done)
            print("info : ", info)

        if done:
            print("Episode is done")
            break

print('finished')


env.close()
