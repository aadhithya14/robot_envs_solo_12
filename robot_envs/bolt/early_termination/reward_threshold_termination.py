class RewardThresholdTermination():

    def __init__(self, robot, threshold, reward_part=None):
        self.robot = robot
        self.threshold = threshold
        self.reward_part = reward_part

    def reset(self):
        pass

    def step(self):
        pass

    def done(self):
        if self.reward_part is not None:
            value = self.robot.rewards[self.reward_part]
        else:
            value = self.robot.reward

        return value < self.threshold
