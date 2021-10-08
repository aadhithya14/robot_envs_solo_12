class RewardAdaptation():

    def __init__(self, robot, conf):
        self.robot = robot
        self.conf = conf
        self.episode_num = 0

    def step(self):
        pass

    def reset(self):
        self.episode_num += 1

        for reward in ['trajectory_tracking_reward', 'velocity_tracking_reward']:
            if reward in self.conf:
                if reward in self.robot.reward_parts:
                    updated_k = min(self.episode_num / self.conf['max_eps'], 1.0) * self.conf[reward]['max_k']
                    self.robot.reward_parts[reward].params['k'] = updated_k

        for reward in ['impact_penalty', 'torque_smoothness_penalty']:
            if reward in self.conf:
                if reward in self.robot.reward_parts:
                    updated_k = min(self.episode_num / self.conf['max_eps'], 1.0) * self.conf[reward]['max_k']
                    self.robot.reward_parts[reward].k = updated_k

        if 'base_y_ang_range' in self.conf:
            current_max_angle = min(self.episode_num / self.conf['max_eps'], 1.0) * self.conf['base_y_ang_range']['max_angle']
            current_termination_angle = current_max_angle + self.conf['base_y_ang_range']['termination_range']
            if 'box_in_air' in self.robot.initialization_conf:
                self.robot.initialization_conf['box_in_air']['base_y_ang_range'] = [-current_max_angle, current_max_angle]
            if 'base_stability_termination' in self.robot.termination_dict:
                self.robot.termination_dict['base_stability_termination'].max_angle = current_termination_angle

        if 'base_x_ang_range' in self.conf:
            current_max_angle = min(self.episode_num / self.conf['max_eps'], 1.0) * self.conf['base_x_ang_range']['max_angle']
            current_termination_angle = current_max_angle + self.conf['base_x_ang_range']['termination_range']
            if 'box_in_air' in self.robot.initialization_conf:
                self.robot.initialization_conf['box_in_air']['base_x_ang_range'] = [-current_max_angle, current_max_angle]
            if 'base_stability_termination' in self.robot.termination_dict:
                if 'base_y_ang_range' in self.conf:
                    if current_termination_angle > self.robot.termination_dict['base_stability_termination'].max_angle:
                        self.robot.termination_dict['base_stability_termination'].max_angle = current_termination_angle
                else:
                    self.robot.termination_dict['base_stability_termination'].max_angle = current_termination_angle

        if 'box_in_air' in self.conf:
            box_in_air = self.conf['box_in_air']
            x_pos = min(self.episode_num / self.conf['max_eps'], 1.0) * box_in_air['max_x_pos']
            y_pos = min(self.episode_num / self.conf['max_eps'], 1.0) * box_in_air['max_y_pos']
            x_ang = min(self.episode_num / self.conf['max_eps'], 1.0) * box_in_air['max_x_ang']
            z_ang = min(self.episode_num / self.conf['max_eps'], 1.0) * box_in_air['max_z_ang']

            init_conf = self.robot.initialization_conf['box_in_air']
            init_conf['base_x_pos_range'] = [-x_pos, x_pos]
            init_conf['base_y_pos_range'] = [-y_pos, y_pos]
            init_conf['base_x_ang_range'] = [-x_ang, x_ang]
            init_conf['base_z_ang_range'] = [-z_ang, z_ang]

        if 'base_static_reward' in self.conf:
            k_vel = min(self.episode_num / self.conf['max_eps'], 1.0) * self.conf['base_static_reward']['max_k_vel']
            k_angvel = min(self.episode_num / self.conf['max_eps'], 1.0) * self.conf['base_static_reward']['max_k_angvel']
            self.robot.reward_parts['base_static_reward'].k_vel = k_vel
            self.robot.reward_parts['base_static_reward'].k_angvel = k_angvel
