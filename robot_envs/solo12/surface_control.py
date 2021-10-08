import os
import numpy as np


class SurfaceControl():

    def __init__(self, robot, surface_type='static_surface', period=0.92, amplitude=0.05, fixed_ground_state=None, fixed_during_episode=False, fixed_amplitude=False, random_initial_position=False):
        self.robot = robot
        self.fixed_ground_state = fixed_ground_state

        urdf_base_string = str(os.path.dirname(os.path.abspath(__file__))) + '/urdf/'
        if surface_type == 'individual_moving_surfaces':
            self.robot.surface_id = self.robot.p.loadURDF(urdf_base_string + "individual_legs.urdf")
        elif surface_type == 'single_moving_surface':
            self.robot.surface_id = self.robot.p.loadURDF(urdf_base_string + "single_surface.urdf")
        elif surface_type == 'tilting_surface':
            self.robot.surface_id = self.robot.p.loadURDF(urdf_base_string + "tilting_surface.urdf")
        elif surface_type == 'y_axis_tilting_surface':
            self.robot.surface_id = self.robot.p.loadURDF(urdf_base_string + "y_axis_tilting_surface.urdf")
        elif surface_type == 'up_down_ytilt_surface':
            self.robot.surface_id = self.robot.p.loadURDF(urdf_base_string + "up_down_ytilt_surface.urdf")
        elif surface_type == 'individual_leg_surfaces':
            self.robot.surface_id = self.robot.p.loadURDF(urdf_base_string + "individual_leg_surfaces.urdf")
        else:
            assert surface_type == 'static_surface', 'Unknown surface type: ' + surface_type
            self.robot.surface_id = self.robot.p.loadURDF(urdf_base_string + "plane_with_restitution.urdf") 
        self.ndof = self.robot.p.getNumJoints(self.robot.surface_id)
      
        if isinstance(period, list):
            self.period = period
        else:
            self.period = self.ndof * [period]

        if isinstance(amplitude, list):
            self.amplitude = amplitude
        else:
            self.amplitude = self.ndof * [amplitude]

        if isinstance(fixed_during_episode, list):
            self.fixed_during_episode = fixed_during_episode
        else:
            self.fixed_during_episode = self.ndof * [fixed_during_episode]

        if isinstance(fixed_amplitude, list):
            self.fixed_amplitude = fixed_amplitude
        else:
            self.fixed_amplitude = self.ndof * [fixed_amplitude]
        

        self.random_initial_position = random_initial_position
        self.surface_type = surface_type
    

    def reset(self):

        if self.surface_type == 'individual_leg_surfaces':
            self.t = 0.0
            self.tdes = np.zeros(4)
            self.xdes = np.zeros(4)
            self.vdes = np.zeros(4)

            up_down_joint_ids = [2, 5, 8, 11]

            for i in range(4):

                self.xdes[i] = np.random.uniform(-self.amplitude[i], self.amplitude[i])
                self.tdes[i] = self.period[i]

                if self.random_initial_position:
                    joint_pos = np.random.uniform(-self.amplitude[i], self.amplitude[i])
                    self.robot.p.resetJointState(
                        bodyUniqueId=self.robot.surface_id,
                        jointIndex=up_down_joint_ids[i],
                        targetValue=joint_pos)
                else:
                    self.robot.p.resetJointState(
                        bodyUniqueId=self.robot.surface_id,
                        jointIndex=up_down_joint_ids[i],
                        targetValue=0.0)

                x = self.robot.p.getJointState(self.robot.surface_id, up_down_joint_ids[i])[0]
                self.vdes[i] = (self.xdes[i] - x) / (self.tdes[i] - self.t)


            return
        
        if self.fixed_ground_state:
            for joint_id in range(self.ndof):
                self.robot.p.resetJointState(
                    bodyUniqueId=self.robot.surface_id,
                    jointIndex=joint_id,
                    targetValue=self.fixed_ground_state[joint_id])
        else:
            self.t = 0.0
            self.tdes = np.zeros(self.ndof)
            self.xdes = np.zeros(self.ndof)
            self.vdes = np.zeros(self.ndof)
            for i in range(self.ndof):
                if self.fixed_during_episode[i]:
                    joint_pos = np.random.uniform(-self.amplitude[i], self.amplitude[i])
                    self.robot.p.resetJointState(
                        bodyUniqueId=self.robot.surface_id,
                        jointIndex=i,
                        targetValue=joint_pos)
                else:
                    if not self.fixed_amplitude[i]:
                        self.xdes[i] = np.random.uniform(-self.amplitude[i], self.amplitude[i])
                    else:
                        self.xdes[i] = self.amplitude[i]
                    self.tdes[i] = self.period[i]
                    # Assuming x = 0 at start for each joint
                    self.vdes[i] = self.xdes[i] / (self.tdes[i] - self.t)

                    self.robot.p.resetJointState(
                        bodyUniqueId=self.robot.surface_id,
                        jointIndex=i,
                        targetValue=0.0)


    def step(self):
        
        if self.surface_type == 'individual_leg_surfaces':


            contacts = self.robot.p.getContactPoints(bodyB=self.robot.robot_id)
            # print(contacts)
            # if contacts is None or len(contacts) == 0:
            endeff_pos, endeff_vel = self.robot.get_endeff_state()
            ok = True
            for i in [2, 5, 8, 11]:
                if endeff_pos[i] < 0.03:
                    ok = False


            limit = 0.01

            for i in [2, 5, 8, 11]:
                joint_z_pos = self.robot.p.getJointState(self.robot.surface_id, i)[0]
                if endeff_pos[i] > joint_z_pos + limit:
                    for j in [1, 2]:
                        self.robot.p.resetJointState(
                            bodyUniqueId=self.robot.surface_id,
                            jointIndex=i - j,
                            targetValue=endeff_pos[i - j])
            #
            # if endeff_pos[2] > 0.03:
            #
            #
            #     # for i in [0, 1, 3, 4, 6, 7, 9, 10]:
            #     #     self.robot.p.setJointMotorControl2(
            #     #         bodyUniqueId=self.robot.surface_id,
            #     #         jointIndex=i,
            #     #         controlMode=self.robot.p.POSITION_CONTROL,
            #     #         targetPosition=endeff_pos[i])
            #
            #     for i in [0, 1, 3, 4, 6, 7, 9, 10]:
            #         self.robot.p.resetJointState(
            #             bodyUniqueId=self.robot.surface_id,
            #             jointIndex=i,
            #             targetValue=endeff_pos[i])


            self.t += self.robot.sim_timestep
            up_down_joint_ids = [2, 5, 8, 11]
            for i in range(4):
                if self.t >= self.tdes[i]:
                    self.xdes[i] = np.random.uniform(-self.amplitude[i], self.amplitude[i])
                    self.tdes[i] += self.period[i]
                    x = self.robot.p.getJointState(self.robot.surface_id, up_down_joint_ids[i])[0]
                    self.vdes[i] = (self.xdes[i] - x) / (self.tdes[i] - self.t)
                self.robot.p.setJointMotorControl2(
                    bodyUniqueId=self.robot.surface_id,
                    jointIndex=up_down_joint_ids[i],
                    controlMode=self.robot.p.VELOCITY_CONTROL,
                    targetVelocity=self.vdes[i])

            return
      


        if not self.fixed_ground_state:
            self.t += self.robot.sim_timestep

            for i in range(self.ndof):
                if not self.fixed_during_episode[i]:
                    if self.t >= self.tdes[i]:
                        if not self.fixed_amplitude[i]:
                            self.xdes[i] = np.random.uniform(-self.amplitude[i], self.amplitude[i])
                        else:
                            self.xdes[i] *= -1.0

                        self.tdes[i] += self.period[i]

                        x = self.robot.p.getJointState(self.robot.surface_id, i)[0]
                        self.vdes[i] = (self.xdes[i] - x) / (self.tdes[i] - self.t)

                    
                    self.robot.p.setJointMotorControl2(
                        bodyUniqueId=self.robot.surface_id,
                        jointIndex=i,
                        controlMode=self.robot.p.VELOCITY_CONTROL,
                        targetVelocity=self.vdes[i])
