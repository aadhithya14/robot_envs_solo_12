import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.plotting import prepare_plot
from utils.experiment_set import ExperimentSet
from robot_envs.hopper.run_policy import run_episode

folder = '185_biped_forward_motion_demo_tracking'

exp_folder = '199'

ep_rewards, list_of_logs, saliency, _, _ = run_episode(
    '/Users/miroslav/work/experiment_results/' + folder + '/' + exp_folder + '/',
    100,
    visualize=True,
    initialization_conf={'on_demo_traj': True, 'use_demo_vel': True, 'fixed_start_timestep': 1000},
    termination_conf={},
    use_movable_disk_surface=True)
