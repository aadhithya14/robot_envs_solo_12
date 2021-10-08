import os
import numpy as np
from robot_envs.bolt.bolt_util import get_foot_locations, get_foot_location_clusters


class MovableDisksSurface():

    def __init__(self,
                 robot,
                 circle_size=0.025,
                 disk_placements_file=None,
                 disk_placements_in_exp_folder=False,
                 height_variation=None,
                 num_fixed_initial_disks=0):
                 
        self.robot = robot
        self.circle_size = circle_size
        self.height_variation = height_variation
        self.num_fixed_initial_disks = num_fixed_initial_disks

        foot_disks_file = self.robot.demo_traj_file[:-5] + 'foot_disks'
        print(foot_disks_file)
        traj_directory = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) \
                         + '/trajectories/'

        if disk_placements_file is not None:
            if disk_placements_in_exp_folder:
                env_name = self.robot.exp_name
                exp_folder = env_name[:env_name.rfind('/') + 1]
                self.clusters = np.loadtxt(exp_folder + disk_placements_file).tolist()
            else:
                self.clusters = np.loadtxt(traj_directory + disk_placements_file).tolist()
        elif os.path.exists(traj_directory + foot_disks_file):
            self.clusters = np.loadtxt(traj_directory + foot_disks_file).tolist()
        else:
            foot_locations = get_foot_locations(self.robot, self.robot.demo_traj)
            self.clusters = get_foot_location_clusters(foot_locations, self.circle_size)

        print(self.clusters)

        self.build_urdf()

    def build_urdf(self):
        current_folder = str(os.path.dirname(os.path.abspath(__file__)))
        with open(current_folder + '/main_file.urdf', 'r') as f:
            main_file = f.read()
        with open(current_folder + '/disk_template.urdf', 'r') as f:
            disk_template = f.read()

        disks = ''
        for i, c in enumerate(self.clusters):
            current_template = disk_template
            current_template = current_template.replace('disk_number', str(i).zfill(3))
            current_template = current_template.replace('x_position', str(c[0]))
            current_template = current_template.replace('y_position', str(c[1]))
            disks = disks + current_template
            # replace joint_name
            # replace link_name
            # replace x_position
            # replace y_position
        # concat all disk templates
        # replace <place_for_joints/> in main_file.urdf with that
        main_file = main_file.replace('<place_for_joints/>', disks)
        # print(main_file)
        with open(current_folder + '/prepared_file.urdf', 'w') as f:
            f.write(main_file)

        self.movable_disks_id = self.robot.p.loadURDF(current_folder + '/prepared_file.urdf')
        self.robot.surface_id = self.movable_disks_id
        self.num_joints = len(self.clusters)
    
    
    def reset(self):
        if self.height_variation is not None:
            for disk_id in range(self.num_fixed_initial_disks, self.num_joints):
                self.robot.p.resetJointState(
                    bodyUniqueId=self.movable_disks_id,
                    jointIndex=disk_id,
                    targetValue=np.random.uniform(-self.height_variation, self.height_variation),
                    targetVelocity=0.0)
