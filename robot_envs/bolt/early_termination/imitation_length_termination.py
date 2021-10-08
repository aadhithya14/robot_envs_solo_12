class ImitationLengthTermination():

    def __init__(self, robot):
        self.robot = robot

    def reset(self):
        # For Solo8 this is set in Solo8 init
        # Having it here might break Solo8
        #self.current_timestep = self.robot.demo_traj_start_timestep
        pass

    def step(self):
        # Assuming here that the demonstration trajectory
        # is given at simulation frequency.
        self.current_timestep += self.robot.cont_timestep_mult

    def done(self):
        # DONE (does not really matter): Check for +-1 error here.
        return self.current_timestep >= (self.robot.demo_traj.shape[0]-1)
