import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import gym
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
from stable_baselines import deepq
from stable_baselines import DQN
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
import datetime
from stable_baselines.deepq.policies import MlpPolicy

from stable_baselines import DQN, PPO2, DDPG
import datetime
# from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.ddpg.policies import DDPGPolicy,MlpPolicy

env = KukaGymEnv(renders=True, isDiscrete=False)
model = DDPG.load("kukadiversecont_model.pkl", env=env)

obs = env.reset()
n = 0
success = 0
while n<100:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    success += rewards
    print(success)
    n =+1
    if dones == True:
        env.reset()
    env.render()

suc = float(success/n)
print('Grasp success: %d' % suc)

