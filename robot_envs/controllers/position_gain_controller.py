import numpy as np
from gym.spaces import Box
from utils.my_math import scale
from utils.data_logging import SimpleLog
import matplotlib.pyplot as plt

def angle_diff(pos, des_pos):
        if des_pos > pos:
            if des_pos - pos < np.pi:
                return des_pos - pos
            else:
                return des_pos - 2 * np.pi - pos
        else:
            if pos - des_pos < np.pi:
                return des_pos - pos
            else:
                return des_pos + 2 * np.pi - pos


class PositionGainController():

    def __init__(self, robot, params, grav_comp=False, robot_log=None):
        self.robot = robot
        self.params = params
        if 'base_kp' in params:
            self.base_kp = np.array(params['base_kp'])
        if 'base_kv' in params:
            self.base_kv = np.array(params['base_kv'])
        self.grav_comp = grav_comp
        self.variant = params['variant']
        self.sqrt_kv_scaling = True
        if 'sqrt_kv_scaling' in params:
            self.sqrt_kv_scaling = params['sqrt_kv_scaling']

        if self.variant in ['vary_single', 'vary_all']:
            self.scaling_base = 10.0
            if 'max_scale' in params:
                self.scaling_base = params['max_scale']
            if 'scaling_range' in params:
                self.scaling_base = np.sqrt(params['scaling_range'])

        self.ndof = self.robot.num_cont_joints
        self.max_torque = self.robot.max_torque
        self.joint_limits = self.robot.joint_limits

        self.log = SimpleLog()
        self.robot_log = robot_log

        self.robot.init_torque_control()

        if 'interp' in self.params and self.params['interp']:
            self.interp_cnt  = 0
            self.poly_params = None

        if self.variant == 'ffqdotdot':
            self.interp_cnt = 0
            self.q0         = 0.0
            self.qdot0      = 0.0

    def get_control_space(self):
        if self.variant == 'fixed':
            return Box(-1.0, 1.0, (self.ndof,))
        elif self.variant == 'vary_single':
            return Box(-1.0, 1.0, (self.ndof + 1,))
        elif self.variant == 'vary_all':
            return Box(-1.0, 1.0, (2 * self.ndof,))
        elif self.variant == 'vary_independent':
            return Box(-1.0, 1.0, (3 * self.ndof,))
        elif self.variant == 'vgdc':
            return Box(-1.0, 1.0, (2 * self.ndof,))
        elif self.variant == 'vgff':
            return Box(-1.0, 1.0, (2 * self.ndof,))
        elif self.variant == 'vg':
            return Box(-1.0, 1.0, (2 * self.ndof,))
        elif self.variant == 'fffg':
            return Box(-1.0, 1.0, (2 * self.ndof,))
        elif self.variant == 'ffvglin':
            return Box(-1.0, 1.0, (3 * self.ndof,))
        elif self.variant == 'ffvgexp':
            return Box(-1.0, 1.0, (3 * self.ndof,))
        elif self.variant == 'ffqqdot':
            return Box(-1.0, 1.0, (3 * self.ndof,))
        elif self.variant == 'ffqdotdot':
            return Box(-1.0, 1.0, (2 * self.ndof,))
        else:
            assert False, 'Unknown PG controller variant: ' + self.variant

    def pos_diff(pos, des_pos, joint_range_type):
        pos_diff = np.zeros(pos.shape)
        for i in range(pos.shape[0]):
            if joint_range_type[i] == 'limited':
                pos_diff[i] = des_pos[i] - pos[i]
            else:
                pos_diff[i] = angle_diff(pos[i], des_pos[i])
        return pos_diff

    def reset(self):
        self.log.clear()


    def vgdc_act(self, action):
        # example parametrization:
        # kp_fixed     :    5.00
        # kp_var_range : [  1.00, 10.00 ]
        # kd_var_range : [  0.05,  0.14 ]

        action_qdes = action[:self.ndof]
        action_gain = action[self.ndof:]

        qdes     = scale(action_qdes, [-1.0, 1.0], self.joint_limits)
        kp_fixed = self.params['kp_fixed']

        if 'kp_var_range' in self.params:
            kp_var = scale(action_gain, [-1.0, 1.0], self.params['kp_var_range'])
            kd_var = scale(action_gain, [-1.0, 1.0], self.params['kd_var_range'])
        else:
            mult = np.power(self.params['pow_base'], action_gain)
            kp_var = mult * self.params['kp_var_mid']
            kd_var = np.sqrt(mult) * self.params['kd_var_mid']

        q, qdot = self.robot.get_cont_joint_state()

        torque = kp_fixed * qdes - kp_var * q - kd_var * qdot

        self.robot.torque_control(torque)


    def vgff_act(self, action):
        # example parametrization
        # kp_var_range: [  1.00, 10.00 ]
        # kd_var_range: [  0.05,  0.14 ]

        action_gain = action[:self.ndof]
        action_ff   = action[self.ndof:]

        ff_torque = scale(action_ff, [-1.0, 1.0], [-self.robot.max_torque, self.robot.max_torque])

        if 'kp_var_range' in self.params:
            kp_var = scale(action_gain, [-1.0, 1.0], self.params['kp_var_range'])
            kd_var = scale(action_gain, [-1.0, 1.0], self.params['kd_var_range'])
        else:
            mult = np.power(self.params['pow_base'], action_gain)
            kp_var = mult * self.params['kp_var_mid']
            kd_var = np.sqrt(mult) * self.params['kd_var_mid']

        q, qdot = self.robot.get_cont_joint_state()

        torque = ff_torque - kp_var * q - kd_var * qdot

        self.robot.torque_control(torque)


    def vg_act(self, action):
        # kp_var (qdes - q) - kd_var qdot

        # example parametrization
        # kp_var_range: [  1.00, 10.00 ]
        # kd_var_range: [  0.05,  0.14 ]

        action_qdes = action[:self.ndof]
        action_gain = action[self.ndof:]

        qdes      = scale(action_qdes, [-1.0, 1.0], self.joint_limits)

        if 'kp_var_range' in self.params:
            kp_var = scale(action_gain, [-1.0, 1.0], self.params['kp_var_range'])
            kd_var = scale(action_gain, [-1.0, 1.0], self.params['kd_var_range'])
        else:
            mult = np.power(self.params['pow_base'], action_gain)
            kp_var = mult * self.params['kp_var_mid']
            kd_var = np.sqrt(mult) * self.params['kd_var_mid']

        q, qdot    = self.robot.get_cont_joint_state()
        joint_type = self.robot.get_cont_joint_type()
        q_diff     = PositionGainController.pos_diff(q, qdes, joint_type)

        torque = kp_var * q_diff - kd_var * qdot

        self.robot.torque_control(torque)


    def fffg_act(self, action):
        # example parametrization
        # kp: 5.0
        # kd: 1.0

        action_ff   = action[:self.ndof]
        action_qdes = action[self.ndof:]

        max_ff_torque = self.robot.max_torque[0]
        if 'max_ff_torque' in self.params:
            max_ff_torque = self.params['max_ff_torque']

        ff_torque = scale(action_ff, [-1.0, 1.0], [-max_ff_torque, max_ff_torque])
        qdes      = scale(action_qdes, [-1.0, 1.0], self.joint_limits)
        # TODO: Temporary, for external access.
        self.des_pos = qdes

        kp        = self.params['kp']
        kd        = self.params['kd']

        q, qdot    = self.robot.get_cont_joint_state()
        joint_type = self.robot.get_cont_joint_type()
        q_diff     = PositionGainController.pos_diff(q, qdes, joint_type)

        p_torque = kp * q_diff
        d_torque = -kd * qdot
        pd_torque = p_torque + d_torque

        if 'max_pd_torque' in self.params:
            pd_torque = np.clip(pd_torque, -self.params['max_pd_torque'], self.params['max_pd_torque'])

        torque = ff_torque + pd_torque

        # self.robot.log.add('action_ff', action_ff)
        # self.robot.log.add('ff_torque', ff_torque)
        # self.robot.log.add('p_torque', p_torque)
        # self.robot.log.add('d_torque', d_torque)
        # self.robot.log.add('pd_torque', pd_torque)
        # self.robot.log.add('torque', torque)

        self.robot.torque_control(torque)


    def ffqqdot_act(self, action):
        # example parametrization
        # kp: 5.0
        # kd: 1.0

        action_ff   = action[:self.ndof]
        action_qdes = action[self.ndof: 2 * self.ndof]
        action_qdotdes = action[2 * self.ndof:]

        max_ff_torque = self.robot.max_torque[0]
        if 'max_ff_torque' in self.params:
            max_ff_torque = self.params['max_ff_torque']

        qdot_range = self.params['qdot_range']

        ff_torque    = scale(action_ff,      [-1.0, 1.0], [-max_ff_torque, max_ff_torque])
        self.qdes    = scale(action_qdes,    [-1.0, 1.0], self.joint_limits)
        self.qdotdes = scale(action_qdotdes, [-1.0, 1.0], qdot_range)

        # No need for copy(), if it is changed it will be overwriten again
        self.qdes_for_ttr = self.qdes
        self.qdotdes_for_ttr = self.qdotdes

        if 'kp_range' in self.params:
            kp = scale(action_ff, [-1.0, 1.0], [self.params['kp_range'][0], self.params['kp_range'][1]])
        else:
            kp = self.params['kp']
        kd = self.params['kd']

        self.q, self.qdot = self.robot.get_cont_joint_state()

        if 'interp' in self.params and self.params['interp']:

            t = self.interp_cnt * self.robot.sim_timestep
            if self.poly_params is not None:
                self.qdes_for_ttr    = self.poly_params.T.dot(np.array([1, t, t ** 2,     t ** 3]))
                self.qdotdes_for_ttr = self.poly_params.T.dot(np.array([0, 1,  2 * t, 3 * t ** 2]))

            if self.interp_cnt == 0:
                # Updating polynomial
                dt = self.robot.cont_timestep_mult * self.robot.sim_timestep

                A = np.array([[1,  0,       0,           0],
                              [0,  1,       0,           0],
                              [1, dt, dt ** 2,     dt ** 3],
                              [0,  1,  2 * dt, 3 * dt ** 2]])
                b = np.array([self.q, self.qdot, self.qdes, self.qdotdes])

                self.poly_params = np.linalg.inv(A).dot(b)

            self.qdes    = self.poly_params.T.dot(np.array([1, t, t ** 2,     t ** 3]))
            self.qdotdes = self.poly_params.T.dot(np.array([0, 1,  2 * t, 3 * t ** 2]))

            self.interp_cnt += 1
            if self.interp_cnt == self.robot.cont_timestep_mult:
                self.interp_cnt = 0

        joint_type = self.robot.get_cont_joint_type()
        q_diff     = PositionGainController.pos_diff(self.q, self.qdes, joint_type)

        p_torque  = kp * q_diff
        d_torque  = kd * (self.qdotdes - self.qdot)
        pd_torque = p_torque + d_torque

        if 'max_pd_torque' in self.params:
            pd_torque = np.clip(pd_torque, -self.params['max_pd_torque'], self.params['max_pd_torque'])

        if 'only_pd' in self.params and self.params['only_pd']:
            torque = pd_torque
        elif 'only_ff' in self.params and self.params['only_ff']:
            torque = ff_torque
        elif 'only_p' in self.params and self.params['only_p']:
            torque = p_torque + kd * (-self.qdot)
        elif 'only_ffp' in self.params and self.params['only_ffp']:
            torque = ff_torque + p_torque + kd * (-self.qdot)
        else:
            torque = ff_torque + pd_torque


        self.robot.log.add('action_ff', action_ff.tolist())
        #self.robot.log.add('ff_torque', ff_torque)
        self.robot.log.add('q',self.q.tolist())
        self.robot.log.add('q_des',self.qdes.tolist())
        self.robot.log.add('p_torque', p_torque.tolist())
        self.robot.log.add('d_torque', d_torque.tolist())
        self.robot.log.add('pd_torque', pd_torque.tolist())
        #self.robot.log.add('torque', torque.tolist())

        self.robot.torque_control(torque)


    def ffqdotdot_act(self, action):
        action_ff         = action[:self.ndof]
        action_qdotdotdes = action[self.ndof:]

        max_ff_torque = self.robot.max_torque[0]
        if 'max_ff_torque' in self.params:
            max_ff_torque = self.params['max_ff_torque']

        qdotdotdes_range = self.params['qdotdotdes_range']

        ff_torque       = scale(action_ff,         [-1.0, 1.0], [-max_ff_torque, max_ff_torque])
        self.qdotdotdes = scale(action_qdotdotdes, [-1.0, 1.0], qdotdotdes_range)

        kp = self.params['kp']
        kd = self.params['kd']

        self.q, self.qdot = self.robot.get_cont_joint_state()

        self.robot.log.add('cnt_q', self.q.tolist())
        self.robot.log.add('cnt_qdot', self.qdot.tolist())

        t                    = self.interp_cnt * self.robot.sim_timestep
        self.qdes_for_ttr    = 0.5 * self.qdotdotdes * t ** 2 + self.qdot0 * t + self.q0
        self.qdotdes_for_ttr = self.qdotdotdes * t + self.qdot0

        if self.interp_cnt == 0:
            dt = self.robot.cont_timestep_mult * self.robot.sim_timestep

            self.q0    = self.q.copy()
            self.qdot0 = self.qdot.copy()

            self.robot.log.add('cnt_q0',    self.q0.tolist())
            self.robot.log.add('cnt_qdot0', self.qdot0.tolist())

        self.qdes    = 0.5 * self.qdotdotdes * t ** 2 + self.qdot0 * t + self.q0
        self.qdotdes = self.qdotdotdes * t + self.qdot0

        self.interp_cnt += 1
        if self.interp_cnt == self.robot.cont_timestep_mult:
            self.interp_cnt = 0

        #==================== Same as ffqqdot ====================

        joint_type = self.robot.get_cont_joint_type()
        q_diff     = PositionGainController.pos_diff(self.q, self.qdes, joint_type)

        p_torque  = kp * q_diff
        d_torque  = kd * (self.qdotdes - self.qdot)
        pd_torque = p_torque + d_torque

        if 'max_pd_torque' in self.params:
            pd_torque = np.clip(pd_torque, -self.params['max_pd_torque'], self.params['max_pd_torque'])

        if 'only_pd' in self.params and self.params['only_pd']:
            torque = pd_torque
        elif 'only_ff' in self.params and self.params['only_ff']:
            torque = ff_torque
        else:
            torque = ff_torque + pd_torque

        if 'torq_mult' in self.params:
            torque *= self.params['torq_mult']

        self.robot.log.add('action_ff', action_ff.tolist())
        self.robot.log.add('ff_torque', ff_torque.tolist())
        self.robot.log.add('p_torque', p_torque.tolist())
        self.robot.log.add('d_torque', d_torque.tolist())
        self.robot.log.add('pd_torque', pd_torque.tolist())

        self.robot.log.add('q_simfreq', self.q.tolist())
        self.robot.log.add('qdot_simfreq', self.qdot.tolist())

        self.robot.torque_control(torque)


    def ffvglin_act(self, action):
        # parametrization
        # kp_range, kd_range

        action_ff   = action[:self.ndof]
        action_gain = action[self.ndof: 2 * self.ndof]
        action_qdes = action[2 * self.ndof:]

        ff_torque = scale(action_ff, [-1.0, 1.0], [-self.robot.max_torque[0], self.robot.max_torque[0]])
        qdes      = scale(action_qdes, [-1.0, 1.0], self.joint_limits)
        # TODO: Temporary, for external access.
        self.des_pos = qdes

        kp = scale(action_gain, [-1.0, 1.0], self.params['kp_range'])
        kd = scale(action_gain, [-1.0, 1.0], self.params['kd_range'])

        q, qdot    = self.robot.get_cont_joint_state()
        joint_type = self.robot.get_cont_joint_type()
        q_diff     = PositionGainController.pos_diff(q, qdes, joint_type)

        torque = ff_torque + kp * q_diff - kd * qdot

        self.robot.torque_control(torque)


    def ffvgexp_act(self, action):
        # parametrization
        # pow_base, kp_mid, kd_mid

        action_ff   = action[:self.ndof]
        action_gain = action[self.ndof: 2 * self.ndof]
        action_qdes = action[2 * self.ndof:]

        ff_torque = scale(action_ff, [-1.0, 1.0], [-self.robot.max_torque[0], self.robot.max_torque[0]])
        qdes      = scale(action_qdes, [-1.0, 1.0], self.joint_limits)
        # TODO: Temporary, for external access.
        self.des_pos = qdes

        mult = np.power(self.params['pow_base'], action_gain)
        kp   = mult * self.params['kp_mid']
        kd   = np.sqrt(mult) * self.params['kd_mid']

        q, qdot    = self.robot.get_cont_joint_state()
        joint_type = self.robot.get_cont_joint_type()
        q_diff     = PositionGainController.pos_diff(q, qdes, joint_type)

        torque = ff_torque + kp * q_diff - kd * qdot

        self.robot.torque_control(torque)


    def act(self, action, raw_des_pos_input=False, ff_torque=0.0, no_torque_clipping=False):

        if self.variant == 'vgdc':
            return self.vgdc_act(action)

        if self.variant == 'vgff':
            return self.vgff_act(action)

        if self.variant == 'vg':
            return self.vg_act(action)

        if self.variant == 'fffg':
            return self.fffg_act(action)

        if self.variant == 'ffvglin':
            return self.ffvglin_act(action)

        if self.variant == 'ffvgexp':
            return self.ffvgexp_act(action)

        if self.variant == 'ffqqdot':
            return self.ffqqdot_act(action)

        if self.variant == 'ffqdotdot':
            return self.ffqdotdot_act(action)

        if raw_des_pos_input:
            self.des_pos = action
        else:
            action = np.clip(action, -1.0, 1.0)
            self.des_pos = scale(action[:self.ndof], [-1.0, 1.0], self.joint_limits)

        if self.variant != 'vary_independent':
            if self.variant == 'fixed':
                scaling_factor = np.ones(self.ndof)
            elif self.variant == 'vary_single':
                scaling_factor = np.power(self.scaling_base, action[-1]) * np.ones(self.ndof)
            elif self.variant == 'vary_all':
                scaling_factor = np.power(self.scaling_base, action[self.ndof:])
            else:
                assert False, 'Unknown PositionGainController variant: ' + self.variant

            self.kp = np.multiply(scaling_factor, self.base_kp)
            if self.sqrt_kv_scaling:
                kv = np.multiply(np.sqrt(scaling_factor), self.base_kv)
            else:
                kv = np.multiply(scaling_factor, self.base_kv)
        else:
            kp_range = self.params['kp_range']
            kd_range = self.params['kd_range']
            self.kp = scale(action[self.ndof: 2 * self.ndof], [-1.0, 1.0], kp_range)
            kv = scale(action[2 * self.ndof:], [-1.0, 1.0], kd_range)

        pos, vel = self.robot.get_cont_joint_state()
        joint_type = self.robot.get_cont_joint_type()
        #   print(pos, des_pos)
        pos_diff = PositionGainController.pos_diff(pos, self.des_pos, joint_type)
        pd_torque = np.multiply(self.kp, pos_diff) - np.multiply(kv, vel)

        gc_torque = 0.0
        assert self.grav_comp == False
        #if self.grav_comp:
        #    gc_torque = self.robot.inv_dyn(np.zeros(self.ndof))

        self.log.add('des_pos', self.des_pos)
        self.log.add('pos', pos)
        self.log.add('vel', vel)
        self.log.add('p_torque', np.multiply(self.kp, self.des_pos - pos))
        self.log.add('d_torque', -np.multiply(kv, vel))
        self.log.add('pd_torque', pd_torque)
        self.log.add('gc_torque', gc_torque)

        if self.robot_log is not None:
            self.log.add('pd_des_pos', self.des_pos)
            self.log.add('pd_pos', pos)

        torque = pd_torque + gc_torque + ff_torque
        self.robot.torque_control(torque, no_clipping=no_torque_clipping)

    def visualize(self):
        prepare_plot()
        plt.rcParams["figure.figsize"] = (24, 8)

        fig, axes = plt.subplots(2, self.ndof)

        pos = np.array(self.log.d['pos'])
        des_pos = np.array(self.log.d['des_pos'])
        pd_torque = np.array(self.log.d['pd_torque'])
        gc_torque = np.array(self.log.d['gc_torque'])

        for i in range(self.ndof):
            axes[0, i].set_title('pos[' + str(i) + ']')
            axes[0, i].grid(True)
            axes[0, i].plot(pos[:, i])
            axes[0, i].plot(des_pos[:, i], '--')

            axes[1, i].set_title('torque[' + str(i) + ']')
            axes[1, i].grid(True)
            axes[1, i].plot(pd_torque[:, i])
            axes[1, i].plot(gc_torque[:, i], '--')

        plt.tight_layout()
        plt.show()
