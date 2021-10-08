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
    pusher_pos = np.array(log['endeff_pos'])

    with open(conf_file, 'r') as f:
        conf = yaml.load(f)

    circle_params = conf['env_params'][0]['env_specific_params']['reward_specs']['circle']

    
    #plt.xlim(-1.0, 1.0)
    #plt.ylim(-1.0, 1.0)
    plt.plot(pusher_pos[:, 0], pusher_pos[:, 1])
    circle = plt.Circle(circle_params['center'], circle_params['radius'], fill=False)
    plt.gca().add_artist(circle)
    plt.tight_layout()
    


counter = {}
exps = [24, 66, 68, 129]
#exps = [8, 10, 12, 80, 84, 100, 132, 145]
#prepare_plot()
#plt.grid(True)
#plt.axis('equal')
for i in range(240):
    try:
    #os.system('python apollo_wall_pushing.py --video ' + folder + str(i).zfill(3) + '/ --minus 0')
    #video_file = None
    #for f in os.listdir(folder + str(i).zfill(3) + '/'):
    #    if f.endswith('.mp4') and not 'tmp' in f:
    #        video_file = f
    #if video_file is not None:
    #    os.system('cp ' + folder + str(i).zfill(3) + '/' + video_file + ' ' + out_folder + '/' + str(i).zfill(3) + '.mp4')
        train_file = folder + str(i).zfill(3) + '/gen_train_log.json'
        with open(train_file, 'r') as f:
            train = json.load(f)

        conf_file = folder + str(i).zfill(3) + '/conf.yaml'
        with open(conf_file, 'r') as f:
            conf = yaml.load(f)
        if i in exps:
            print(conf['env_params'][0]['env_specific_params']['reward_specs']['circle'])
            print(conf['rnn_noise_network'])
            print(conf['env_params'][0]['max_episode_steps'])
            print('---')


        #plt.plot(train['validation_rewards'])
        #print(train.keys())
        #visualize(log_file, conf_file, plot_file)
    except:
        pass
#plt.show()

    