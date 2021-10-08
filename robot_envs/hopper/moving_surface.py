import pybullet as p
import time
import pybullet_data
import os
import numpy as np

class MovingSurface():

    def __init__(self, p, dt, joint_index=1, period=0.739, amplitude=0.1, small_moving_surface=False):
        self.p = p
        self.dt = dt
        self.joint_index = joint_index
        self.period = period
        self.amplitude = amplitude
        urdf_base_string = str(os.path.dirname((os.path.abspath(__file__))))
        if small_moving_surface:
            self.surface_id = self.p.loadURDF(urdf_base_string + "/small_moving_surface.urdf")
        else:
            self.surface_id = self.p.loadURDF(urdf_base_string + "/moving_surface.urdf")

    def reset(self):
        self.t = 0.0
        self.xdes = np.random.uniform(-self.amplitude, self.amplitude)
        self.tdes = self.period
        self.p.resetJointState(
            bodyUniqueId=self.surface_id,
            jointIndex=self.joint_index,
            targetValue=0.0)

    def step(self):
        self.t += self.dt
        if self.t >= self.tdes:
            self.xdes = np.random.uniform(-self.amplitude, self.amplitude)
            self.tdes += self.period
        x = self.p.getJointState(self.surface_id, self.joint_index)[0]

        self.p.setJointMotorControl2(
            bodyUniqueId=self.surface_id,
            jointIndex=self.joint_index,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=(self.xdes - x) / (self.tdes - self.t))

if __name__ == '__main__':
    '''
    physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    #p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    p.setGravity(0,0,-10)
    path = os.path.dirname(os.path.abspath(__file__))
    planeId = p.loadURDF("moving_surface.urdf")
    cubeStartPos = [0,0,1]
    cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
    p.resetJointState(
                    bodyUniqueId=planeId,
                    jointIndex=1,
                    targetValue=0.0)
    for i in range (10000):

        p.setJointMotorControl2(
                    bodyUniqueId=planeId,
                    jointIndex=1,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=1.0)
        p.stepSimulation()
        time.sleep(1./240.)
    p.disconnect()
    '''
    p.connect(p.GUI, options='--background_color_red=0.431 --background_color_green=0.856 --background_color_blue=0.546')
    p.resetDebugVisualizerCamera( cameraDistance=1, cameraYaw=0, cameraPitch=-35, cameraTargetPosition=[0,0,0])
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

    dt = 1.0 / 240.0
    moving_surface = MovingSurface(p, dt)
    moving_surface.reset()
    for i in range(10000):
        moving_surface.step()
        p.stepSimulation()
        time.sleep(dt)