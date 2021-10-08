import stable_baselines3
import gym
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
import robot_envs
import gym, logging, gym.envs
import torch
import torch.nn as nn
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import time
from stable_baselines3.common import results_plotter
#from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
#from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import matplotlib.pyplot as plt
import yaml

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'episodes')
    y = moving_average(y, window=100)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.savefig("robot_envs/solo12/experiments/stage1_solo12_jumping_motion/006/plot.png")

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        squash_output=True,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            squash_output=squash_output
            *args,
            **kwargs,
        )
        
    
def train(env_id, num_timesteps, seed, exp_name, exp_config):
    env_params = exp_config['env_params'][0]['env_specific_params']
    env_params['output_dir'] = exp_config['output_dir']
    env_params['exp_name'] = exp_name + 'env'

    if not 'max_episode_steps' in exp_config['env_params'][0]:
        max_episode_duration = exp_config['env_params'][0]['max_episode_duration']
        sim_timestep = exp_config['env_params'][0]['env_specific_params']['sim_timestep']
        cont_timestep_mult = exp_config['env_params'][0]['env_specific_params']['cont_timestep_mult']

        max_episode_steps = int(max_episode_duration / (cont_timestep_mult * sim_timestep))
        exp_config['env_params'][0]['max_episode_steps'] = max_episode_steps

    env_id="Solo12-v0"
    gym.envs.register(
        id=env_id,
        entry_point=exp_config['env_params'][0]['entry_point'],
        max_episode_steps=exp_config['env_params'][0]['max_episode_steps'],
        kwargs=env_params
        )

    env = gym.make(env_id)
    log_dir = "robot_envs/solo12/experiments/stage1_solo12_jumping_motion/006/logdir"
    #env=make_vec_env('Solo12-v0',monitor_dir=log_dir)

    os.makedirs(log_dir, exist_ok=True)
    env=Monitor(env,log_dir)

    hid_size = 64
    if 'hid_size' in exp_config:
        hid_size = exp_config['hid_size']

    num_hid_layers = 2
    if 'num_hid_layers' in exp_config:
        num_hid_layers = exp_config['num_hid_layers']

    net_arch=[hid_size]*num_hid_layers

    optim_stepsize = 3e-4
    if 'optim_stepsize' in exp_config:
        optim_stepsize = exp_config['optim_stepsize']


   
    
# Save a checkpoint every 200000 steps
    checkpoint_callback = CheckpointCallback(save_freq=200000, save_path='./robot_envs/solo12/experiments/stage1_solo12_jumping_motion/006/checkpoints',
                                         name_prefix='model')
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold= 1500.0, verbose=1)
    eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best,eval_freq=1000000)

    policy_kwargs=dict(net_arch=net_arch)
    model= PPO(CustomActorCriticPolicy,env,learning_rate=linear_schedule(optim_stepsize),batch_size=64,n_steps=2048,policy_kwargs=policy_kwargs,verbose=1,n_epochs=10,clip_range=0.2, ent_coef=0.0,gamma=0.99,gae_lambda=0.95,tensorboard_log="./robot_envs/solo12/experiments/stage1_solo12_jumping_motion/006/log/ppo_006/")
    #total_timesteps=5000
    model.learn(total_timesteps=num_timesteps,callback=[checkpoint_callback,eval_callback], tb_log_name="first_run",reset_num_timesteps=False)
    model.save("robot_envs/solo12/experiments/stage1_solo12_jumping_motion/006/model")

    reward_list=env.get_episode_rewards()
    mean_reward_list=[]
    for i in range(len(reward_list)):
        mean_reward=np.mean(reward_list[:i])
        mean_reward_list.append(mean_reward)
    
    plt.plot(mean_reward_list)
    plt.xlabel("episodes")
    plt.ylabel("mean reward per episode")
    plt.savefig("robot_envs/solo12/experiments/stage1_solo12_jumping_motion/006/mean_reward.png")

    
    plt.plot(reward_list)
    plt.xlabel("episodes")
    plt.ylabel("reward per episode")
    plt.savefig("robot_envs/solo12/experiments/stage1_solo12_jumping_motion/006/reward.png")
    plot_results(log_dir)

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Solo12-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--name', help='experiment name', default='unnamed')
    parser.add_argument('--exp', help='')
    args = parser.parse_args()

    exp_config = yaml.load(open(args.exp + 'conf.yaml'))

    seed = args.seed
    if 'seed' in exp_config:
        seed = exp_config['seed']

    train(args.env, num_timesteps=exp_config['training_timesteps'], seed=seed, exp_name=args.exp, exp_config=exp_config)


if __name__ == '__main__':
    main()



    
    



