#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from pybullet_envs.bullet.tm700GymEnv import tm700GymEnv
from pybullet_envs.bullet.kukaCamGymEnv import KukaCamGymEnv
from stable_baselines import deepq
from stable_baselines import DQN, PPO2, DDPG
import datetime
# from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.ddpg.policies import DDPGPolicy,MlpPolicy

def callback(lcl, glb):
  # stop training if reward exceeds 199
  total = sum(lcl['episode_rewards'][-101:-1]) / 100
  totalt = lcl['t']
  #print("totalt")
  #print(totalt)
  is_solved = totalt > 2000 and total >= 10
  return is_solved


def main():

  env1 = tm700GymEnv(renders=True, isDiscrete=False)
  model = DDPG(MlpPolicy, env1, verbose=1)

  # = deepq.models.mlp([64])
  model.learn(total_timesteps=500000)
                    #max_timesteps=10000000,
                    # exploration_fraction=0.1,
                    # exploration_final_eps=0.02,
                    # print_freq=10,
                    # callback=callback, network='mlp')
  print("Saving model to kukadiverse_model.pkl")
  model.save("kukadiversecont_model.pkl")


  main()

