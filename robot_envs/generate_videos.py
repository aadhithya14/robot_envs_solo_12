import os

#for exp in experiments
#latest
folder = '~/work/experiments_cluster/115_rnn_circle_baseline/'
#os.system('python apollo_wall_pushing.py --video ' + folder + '000/ --minus 1')
# python apollo_wall_pushing.py --video ~/work/experiments_cluster/115_rnn_circle_baseline/000/ --minus 0


for i in range(8, 240):
	os.system('python apollo_wall_pushing.py --video ' + folder + str(i).zfill(3) + '/ --minus 1')