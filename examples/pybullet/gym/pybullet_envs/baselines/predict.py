import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import gym
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
from pybullet_envs.bullet.tm700GymEnv_TEST import tm700GymEnv2

from stable_baselines import deepq
from stable_baselines import DQN
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
import datetime
from stable_baselines.deepq.policies import MlpPolicy

from stable_baselines import DQN, PPO2, DDPG
import datetime
# from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.ddpg.policies import DDPGPolicy,MlpPolicy
import time

env = tm700GymEnv2(renders=True, isDiscrete=False)
model = DDPG.load("tm_test_model.pkl", env=env)

obs = env.reset()
n = 0
success = 0
# while n<1000:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     success += rewards
#     n =+1
#     if dones == True:
#         env.reset()
#     env.render()

# suc = float(success/n)
# print('Grasp success: %d' % suc)
time_step_counter = 0
iterations = 20000
while time_step_counter < iterations:
    action, _ = model.predict(obs)
    obs, rewards, dones, _ = env.step(action)  # Assumption: eval conducted on single env only!
    time_step_counter +=1

    # time.sleep(0.1)
    if dones:
        obs = env.reset()
