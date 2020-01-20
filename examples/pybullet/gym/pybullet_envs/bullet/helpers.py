# from stable_baselines.common.env_checker import check_env
from pybullet_envs.bullet.tm700GymEnv_TEST import tm700GymEnv2

env = tm700GymEnv2()
# It will check your custom environment and output additional warnings if needed
# check_env(env)

obs = env.reset()
n_steps = 10
for _ in range(n_steps):
    # Random action
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)