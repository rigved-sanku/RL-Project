"""
Sample environment for simulating quadrotor

"""
import sys
import site

# PATH CONFIGURATION
user_site_packages =site.getusersitepackages()
sys.path.append(user_site_packages) #For pip installed dependencies
sys.path.append('./simulator')

# IMPORT PIP LIBS
import importlib
import numpy as np
import scipy
    
# IMPORT DYNAMICS, CONTROL and USER CODE
import quad_dynamics as qd
import control
import tello
import rendering
import usercode

# Force reload custom modules and run the latest code
importlib.reload(control)
importlib.reload(qd)
importlib.reload(tello)
importlib.reload(rendering)
importlib.reload(usercode)

class Env():
    def __init__(self):
        # CONSTANTS
        self.fps = 20

        # STOP time for simulation
        self.sim_stop_time = 15

        # INIT RENDERING AND CONTROL
        self.controller = control.quad_control()
        self.user_sm = usercode.state_machine()

        # SET TIME STEP
        self.dynamics_dt = 0.01
        self.control_dt = self.controller.dt
        self.user_dt = self.user_sm.dt
        self.frame_dt = 1./self.fps

        # INIT TIMER
        self.dynamics_countdown = 0.
        self.control_countdown = 0.
        self.frame_countdown = 0.
        self.user_countdown = 0.
    
    def reset(self):
        # INIT STATES
        self.current_time = 0.
        xyz = np.array([0.0, 0.0, -5.0])
        vxyz = np.array([0.0, 0.0, 0.0])
        quat = np.array([1.0, .0, .0, .0])
        pqr = np.array([0.0, .0, .0])
        self.current_ned_state = np.concatenate((xyz, vxyz, quat, pqr))

        # INIT LOG
        self.stateArray = self.current_ned_state
        self.timeArray = 0
        self.controlArray = np.array([0., 0, 0, 0])

    def step(self):
        if current_time < self.sim_stop_time:

            if user_countdown<=0:
                xyz_ned = current_ned_state[0:3]
                xyz_blender = [xyz_ned[0], -xyz_ned[1], -xyz_ned[2]]

                vxyz_ned = current_ned_state[3:6]
                vxyz_blender = [vxyz_ned[0], -vxyz_ned[1], -vxyz_ned[2]]

                xyz_bl_des, vel_bl_des, acc_bl_des, yaw_bl_setpoint = user_sm.step(current_time, xyz_blender, vxyz_blender)

                yaw_ned = -yaw_bl_setpoint
                WP_ned = np.array([xyz_bl_des[0], -xyz_bl_des[1], -xyz_bl_des[2], yaw_ned])
                vel_ned = np.array([vel_bl_des[0], -vel_bl_des[1], -vel_bl_des[2]])
                acc_ned = np.array([acc_bl_des[0], -acc_bl_des[1], -acc_bl_des[2]])
                
                user_countdown = user_dt

            if control_countdown<=0.:
                U = controller.step(current_ned_state, WP_ned, vel_ned, acc_ned)
                control_countdown = control_dt

            # Dynamics runs at base rate. 
            #   TODO replace it with ODE4 fixed step solver
            current_ned_state = current_ned_state + dynamics_dt*qd.model_derivative(current_time,
                                                                current_ned_state,
                                                                U,
                                                                tello)
            
            # UPDATE COUNTDOWNS AND CURRENT TIME
            dynamics_countdown -= dynamics_dt
            control_countdown -= dynamics_dt
            frame_countdown -= dynamics_dt
            user_countdown -= dynamics_dt
            current_time += dynamics_dt

            # LOGGING
            stateArray = np.vstack((stateArray, current_ned_state))
            controlArray = np.vstack((controlArray, U))
            timeArray = np.append(timeArray, current_time)
        else:


    def log():
        # ----------------------------------------------------------------------------------------------
        user_sm.terminate()

        # SAVE LOGGED SIGNALS TO MAT FILE FOR POST PROCESSING IN MATLAB
        loggedDict = {'time': timeArray,
                    'state': stateArray,
                    'control': controlArray}  
        scipy.io.savemat('./log/states.mat', loggedDict)

    
     