import numpy as np


kp1force = 58.9
kp10force = 71.4

def exp_rew(x, x_range, y_range, curve, flipped=False):
    def g(x):
        return np.exp(curve * x)
    
    
    def f(x):
        return (g(x) - g(0)) / (g(1) - g(0))
        
    
    if flipped:
        return y_range[1] - (f((x_range[1] - x) / (x_range[1] - x_range[0])) * (y_range[1] - y_range[0]))
    else:
        return y_range[0] + f((x - x_range[0]) / (x_range[1] - x_range[0])) * (y_range[1] - y_range[0])


def r1(f):
    if f < 20:
        return exp_rew(f, [0.0, 20], [0.0, 0.2], 0.001)
    else:
        return exp_rew(f, [20, kp10force], [0.2, 4], 5, False)
    
def r2(f):
    if f < 20:
        return exp_rew(f, [0.0, 20], [0.0, 0.2], 0.001)
    elif f < kp1force:
        return exp_rew(f, [20, kp1force], [0.2, 1], 0.001)
    else:
        return exp_rew(f, [kp1force, kp10force], [1, 4], 2, True)
    
def r3(f):
    return exp_rew(f, [0, kp10force], [0, 4.0], 0.001)
    

def r4(f):
    if f < 20:
        return exp_rew(f, [0.0, 20], [0.0, 0.2], 0.001)
    elif f < kp1force:
        return exp_rew(f, [20, kp1force], [0.2, 1], 0.001)
    else:
        return exp_rew(f, [kp1force, kp10force], [1, 4], 0.001)

def r5(f):
    if f < 20:
        return 0.0
    else:
        return exp_rew(f, [20, kp10force], [0.0, 4], 5, False)
    
def r6(f):
    if f < 20:
        return 0.0
    elif f < kp1force:
        return exp_rew(f, [20, kp1force], [0.0, 1], 0.001)
    else:
        return exp_rew(f, [kp1force, kp10force], [1, 4], 2, True)
    
def r7(f):
    if f < 20:
        return 0.0
    else:
        return exp_rew(f, [20, kp10force], [0, 4.0], 0.001)
    

def r8(f):
    if f < 20:
        return 0.0
    elif f < kp1force:
        return exp_rew(f, [20, kp1force], [0.0, 1], 0.001)
    else:
        return exp_rew(f, [kp1force, kp10force], [1, 4], 0.001)

ok_landing = 58.9
max_pen = 100

rew_at_pushoff = 0.1
rew_above_ok_landing = 2.0

def r9(f, pushoff=40.0):
    if f < pushoff:
        return 0.0
    elif f < max_pen:
        return exp_rew(f, [pushoff, max_pen], [0.1, 8], 3, False)
    else:
        return exp_rew(max_pen, [pushoff, max_pen], [0.1, 8], 3, False)
    
def r10(f, pushoff=40.0):
    rew_at_ok_landing = 0.4 * 40.0 / pushoff
    if f < pushoff:
        return 0.0
    elif f < ok_landing:
        return exp_rew(f, [pushoff, ok_landing], [rew_at_pushoff, rew_at_ok_landing], 0.001)
    else:
        return exp_rew(f, [ok_landing, max_pen], [rew_above_ok_landing, 8], 2, True)
    
def r11(f, pushoff=40.0):
    if f < pushoff:
        return 0.0
    else:
        return exp_rew(f, [pushoff, max_pen], [0, 8.0], 0.001)
    

def r12(f, pushoff=40.0):
    rew_at_ok_landing = 0.4 * 40.0 / pushoff
    if f < pushoff:
        return 0.0
    elif f < ok_landing:
        return exp_rew(f, [pushoff, ok_landing], [0.0, rew_at_ok_landing], 0.001)
    else:
        return exp_rew(f, [ok_landing, max_pen], [rew_above_ok_landing, 8], 0.001)
    
def r13(f, pushoff=40.0):
    if f < pushoff:
        return 0.0
    elif f < max_pen:
        return exp_rew(f, [pushoff, max_pen], [0.1, 8], 3, True)
    else:
        return exp_rew(max_pen, [pushoff, max_pen], [0.1, 8], 3, True)


class ForcePenalty():

    def __init__(self, robot, params):
        self.robot = robot
        self.params = params
        if 'des_force' in self.params:
            self.des_force = self.params['des_force']
        else:
            self.des_force = 40.0

    def reset(self):
        pass

    def get_reward(self):
        f = -self.robot.get_endeff_force()[2]
        k = self.params['k']
        if self.params['variant'] == 'r1':
            return -k * r1(f)
        elif self.params['variant'] == 'r2':
            return -k * r2(f)
        elif self.params['variant'] == 'r3':
            return -k * r3(f)
        elif self.params['variant'] == 'r4':
            return -k * r4(f)
        elif self.params['variant'] == 'r5':
            return -k * r5(f)
        elif self.params['variant'] == 'r6':
            return -k * r6(f)
        elif self.params['variant'] == 'r7':
            return -k * r7(f)
        elif self.params['variant'] == 'r8':
            return -k * r8(f)
        elif self.params['variant'] == 'r9':
            return -k * r9(f, self.des_force)
        elif self.params['variant'] == 'r10':
            return -k * r10(f, self.des_force)
        elif self.params['variant'] == 'r11':
            return -k * r11(f, self.des_force)
        elif self.params['variant'] == 'r12':
            return -k * r12(f, self.des_force)
        elif self.params['variant'] == 'r13':
            return -k * r13(f, self.des_force)
        else:
            assert False, 'Unknown ForcePenalty reward variant: ' + self.params['variant']