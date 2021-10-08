import os
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
from utils.plotting import prepare_plot

#folder = '~/work/experiments_cluster/115_rnn_circle_baseline/'
#folder = '/Users/miroslav/work/experiments_cluster/117_ppo_reaching_baseline/'
out_folder = '/Users/miroslav/Desktop/119/'
folder = '/Users/miroslav/work/experiments_cluster/119_rnn_circle_baseline/'

#for exp in experiments
#latest

#os.system('python apollo_wall_pushing.py --video ' + folder + '000/ --minus 1')
# python apollo_wall_pushing.py --video ~/work/experiments_cluster/115_rnn_circle_baseline/000/ --minus 0

def visualize(log_file, conf_file, plot_file):
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


counter = {}
for i in range(240):
    try:
    #os.system('python apollo_wall_pushing.py --video ' + folder + str(i).zfill(3) + '/ --minus 0')
    #video_file = None
    #for f in os.listdir(folder + str(i).zfill(3) + '/'):
    #    if f.endswith('.mp4') and not 'tmp' in f:
    #        video_file = f
    #if video_file is not None:
    #    os.system('cp ' + folder + str(i).zfill(3) + '/' + video_file + ' ' + out_folder + '/' + str(i).zfill(3) + '.mp4')
        conf_file = folder + str(i).zfill(3) + '/conf.yaml'
        latest_log = max(os.listdir(folder + str(i).zfill(3) + '/val_env_episodes/'))
        log_file = folder + str(i).zfill(3) + '/val_env_episodes/' + latest_log
        plot_file = out_folder + str(i).zfill(3) + '_torq.png'
        visualize(log_file, conf_file, plot_file)

    except:
        pass

