import os
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
from utils.plotting import prepare_plot

#folder = '~/work/experiments_cluster/115_rnn_circle_baseline/'
#folder = '/Users/miroslav/work/experiments_cluster/117_ppo_reaching_baseline/'
out_folder = '/home/miroslav/work/videos/122_rnn_circle_baseline_vel_pen/'
folder = '/home/miroslav/cluster_home_002/experiments/122_rnn_circle_baseline_vel_pen/'

#for exp in experiments
#latest

#os.system('python apollo_wall_pushing.py --video ' + folder + '000/ --minus 1')
# python apollo_wall_pushing.py --video ~/work/experiments_cluster/115_rnn_circle_baseline/000/ --minus 0

def visualize(log_file, conf_file, plot_file):
    with open(log_file, 'r') as f:
        log = json.load(f)

    print(log.keys())
    pusher_pos = np.array(log['endeff_pos'])

    with open(conf_file, 'r') as f:
        conf = yaml.load(f)

    circle_params = conf['env_params'][0]['env_specific_params']['reward_specs']['circle']

    prepare_plot(wide=False)
    plt.grid(True)
    plt.axis('equal')
    #plt.xlim(-1.0, 1.0)
    #plt.ylim(-1.0, 1.0)
    plt.plot(pusher_pos[:, 0], pusher_pos[:, 1])
    circle = plt.Circle(circle_params['center'], circle_params['radius'], fill=False)
    plt.gca().add_artist(circle)
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()


def plot_velocity(i):
    conf_file = folder + str(i).zfill(3) + '/conf.yaml'
    latest_log = max(os.listdir(folder + str(i).zfill(3) + '/val_env_episodes/'))
    log_file = folder + str(i).zfill(3) + '/val_env_episodes/' + latest_log
    plot_file = out_folder + str(i).zfill(3) + '_vel.png'

    with open(log_file, 'r') as f:
        log = json.load(f)

    print(log.keys())
    joint_vel = np.array(log['joint_vel'])

    with open(conf_file, 'r') as f:
        conf = yaml.load(f)

    prepare_plot()
    plt.grid(True)
    #plt.xlim(-1.0, 1.0)
    #plt.ylim(-1.0, 1.0)
    for i in range(joint_vel.shape[1]):
        plt.plot(joint_vel[:, i], label=str(i))
    plt.legend(frameon=1, framealpha=1.0)
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()


def plot_vel_pen(i):
    conf_file = folder + str(i).zfill(3) + '/conf.yaml'
    latest_log = max(os.listdir(folder + str(i).zfill(3) + '/val_env_episodes/'))
    log_file = folder + str(i).zfill(3) + '/val_env_episodes/' + latest_log
    plot_file = out_folder + str(i).zfill(3) + '_vel_pen.png'

    with open(log_file, 'r') as f:
        log = json.load(f)

    print(log.keys())
    vel_pen = np.array(log['velocity_penalty'])

    with open(conf_file, 'r') as f:
        conf = yaml.load(f)

    prepare_plot()
    plt.grid(True)
    #plt.xlim(-1.0, 1.0)
    #plt.ylim(-1.0, 1.0)
    plt.plot(vel_pen)
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()


def plot_torque(i):
    conf_file = folder + str(i).zfill(3) + '/conf.yaml'
    latest_log = max(os.listdir(folder + str(i).zfill(3) + '/val_env_episodes/'))
    log_file = folder + str(i).zfill(3) + '/val_env_episodes/' + latest_log
    plot_file = out_folder + str(i).zfill(3) + '_torq.png'

    with open(log_file, 'r') as f:
        log = json.load(f)

    print(log.keys())
    action = np.array(log['latest_action'])

    with open(conf_file, 'r') as f:
        conf = yaml.load(f)

    prepare_plot()
    plt.grid(True)
    #plt.xlim(-1.0, 1.0)
    #plt.ylim(-1.0, 1.0)
    max_torque = 0.1 * np.array([200.0, 200.0, 100.0, 100.0, 100.0, 30.0, 30.0])
    torque = action * max_torque
    for i in range(torque.shape[1]):
        plt.plot(torque[:, i], label=str(i))
    plt.legend(frameon=1, framealpha=1.0)
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()


def plot_learning_curve(i):
    train_log = folder + str(i).zfill(3) + '/gen_train_log.json'
    plot_file = out_folder + str(i).zfill(3) + '_learning_curve.png'

    with open(train_log, 'r') as f:
        log = json.load(f)

 

    prepare_plot()
    plt.grid(True)
    #plt.xlim(-1.0, 1.0)
    #plt.ylim(-1.0, 1.0)
    plt.plot(log['validation_rewards'])
    plt.legend(frameon=1, framealpha=1.0)
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()


def generate_video(i):
    os.system('python apollo_wall_pushing.py --video ' + folder + str(i).zfill(3) + '/ --minus 0')
    video_file = None
    for f in os.listdir(folder + str(i).zfill(3) + '/'):
        if f.endswith('.mp4') and not 'tmp' in f:
            video_file = f
    if video_file is not None:
        os.system('cp ' + folder + str(i).zfill(3) + '/' + video_file + ' ' + out_folder + '/' + str(i).zfill(3) + '.mp4')

counter = {}
for i in range(0, 300):
    '''try:
        generate_video(i)
    except:
        pass
    try:
        plot_velocity(i)
    except:
        pass
    try:
        plot_vel_pen(i)
    except:
        pass
    try:
        plot_torque(i)
    except:
        pass
    try:
        plot_learning_curve(i)
    except:
        pass'''
    try:
        conf_file = folder + str(i).zfill(3) + '/conf.yaml'
        with open(conf_file, 'r') as f:
            conf = yaml.load(f)
        print(i, conf['env_params'][0]['env_specific_params']['reward_specs']['velocity_penalty']['k_v'])
    except:
        pass

    