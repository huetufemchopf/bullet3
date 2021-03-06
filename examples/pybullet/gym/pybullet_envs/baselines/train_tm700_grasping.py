#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
from pybullet_envs.bullet.tm700_diverse_object_gym_env import tm700DiverseObjectEnv
from pybullet_envs.bullet.tm700GymEnv_TEST import tm700GymEnv2
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
from pybullet_envs.bullet.tm700GymEnv import tm700GymEnv
from stable_baselines import deepq
from stable_baselines import DQN, PPO2, DDPG, HER
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
import datetime
# from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.ddpg.policies import DDPGPolicy,MlpPolicy
import time

def callback(lcl, glb):
  # stop training if reward exceeds 199
  total = sum(lcl['episode_rewards'][-101:-1]) / 100
  totalt = lcl['t']
  #print("totalt")
  #print(totalt)
  is_solved = totalt > 2000 and total >= 10
  return is_solved




def main():
  param_noise = None

  env1 = tm700GymEnv2(renders=False , isDiscrete=False)
  model = DDPG(MlpPolicy, env1, verbose=1, param_noise=param_noise, random_exploration=0.1)
  # model = DQN(MlpPolicy, env1, verbose=1, exploration_fraction=0.3)

  # = deepq.models.mlp([64])
  start = time.time()
  model.learn(total_timesteps=1000000)
                    #max_timesteps=10000000,
                    # exploration_fraction=0.1,
                    # exploration_final_eps=0.02,
                    # print_freq=10,
                    # callback=callback, network='mlp')
  print("Saving model")
  model.save("tm_test_model_randomblocksrotated.pkl")

  print('total time', time.time()-start)

def heralgorithm():

    goal_selection_strategy = 'future'  # equivalent to GoalSelectionStrategy.FUTURE

    # Wrap the model
    model = HER('MlpPolicy', env1, DDPG, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,
                verbose=1)
    # Train the model
    model.learn(1000)

    model.save("./her_bit_env")

main()

