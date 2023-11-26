import numpy as np
import math
from pyquaternion import Quaternion
from numpy.linalg import norm

class quad_control:
    def __init__(self):
        # CONTROLLER PROPERTIES AND GAINS
        dt = 0.010
        filter_tau = 0.04
        self.dt = dt

        self.maxRate = 1.5
        maxAct = 0.3

        self.minRate = -self.maxRate
        minAct = -maxAct

        # Angular velocity controller
        kp_angvel = 6.
        self.p_pid = pid(kp_angvel, 0, kp_angvel/15., filter_tau, dt, minVal = minAct, maxVal = maxAct)
        self.q_pid = pid(kp_angvel, 0, kp_angvel/15., filter_tau, dt, minVal = minAct, maxVal = maxAct)
        self.r_pid = pid(kp_angvel, 0, kp_angvel/15, filter_tau, dt, minVal = minAct, maxVal = maxAct)

        # For logging
        self.current_time = 0.
        self.timeArray = 0
        self.controlArray = np.array([0., 0, 0, 0])

    def step(self, X, pqr_sp, throttle):
        # EXTRACT STATES
        pqr = X[10:13]

        # ANGULAR VELOCITY
        tau_x = self.p_pid.step(pqr_sp[0], pqr[0])
        tau_y = self.q_pid.step(pqr_sp[1], pqr[1])
        tau_z = self.r_pid.step(pqr_sp[2], pqr[2])

        # MIXER
        u1 = throttle - tau_x + tau_y + tau_z
        u2 = throttle + tau_x - tau_y + tau_z
        u3 = throttle + tau_x + tau_y - tau_z
        u4 = throttle - tau_x - tau_y - tau_z

        U = np.array([u1, u2, u3, u4])
        U = U.clip(0.0, 1.0)

        # Logger
        self.controlArray = np.vstack((self.controlArray, np.array((throttle, tau_x, tau_y, tau_z))))
        self.timeArray = np.append(self.timeArray, self.current_time)
        self.current_time+=self.dt

        return U

class pid:
    def __init__(self, kp, ki, kd, filter_tau, dt, dim = 1, minVal = -1, maxVal = 1):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.minVal = minVal
        self.maxVal = maxVal
        self.filter_tau = filter_tau
        self.dt = dt

        self.minVal = minVal
        self.maxVal = maxVal

        if dim == 1:
            self.prev_filter_val = 0.0
            self.prev_err = 0.0
            self.prev_integral = 0.0
        else:
            self.prev_err = np.zeros(dim, dtype="double")
            self.prev_filter_val = np.zeros(dim, dtype="double")
            self.prev_integral = np.zeros(dim, dtype="double")
    
    def step(self, dsOrErr, current_state = None):

        # Error
        if current_state is None:
            err = dsOrErr
        else:
            desired_state = dsOrErr
            err = desired_state - current_state

        # Error Derivative and filtering
        err_der = (err-self.prev_err)/self.dt

        # Forward euler discretization of first order LP filter
        alpha = self.dt/self.filter_tau
        err_der_filtered = err_der*alpha + self.prev_filter_val*(1-alpha)

        # Integral
        err_integral = err*self.dt + self.prev_integral

        # Raw Output
        out = self.kp*err + self.kd*err_der_filtered + self.ki*err_integral

        # NaN check
        if math.isnan(out):
            print('err', err)
            print(err_integral)
            print(err_der)
            print('Make sure waypoints are not nan. If you still get this error, contact your TA.')
            if current_state is None:
                print('Error is directly provided to the PID')
            else:
                print('desired - ', desired_state)
                print('current - ', current_state)
            raise Exception('PID blew up :( out is nan')

        # Update the internal states
        self.prev_err = err
        self.prev_filter_val = err_der_filtered
        self.prev_integral = err_integral

        # Integral anti-windup. Clamp values
        self.prev_integral = np.clip(self.prev_integral, self.minVal, self.maxVal)

        # Clip the final output
        out = np.clip(out, self.minVal, self.maxVal)

        # Inf check
        if math.isinf(out):
            raise Exception('PID output is inf')
        
        return out
