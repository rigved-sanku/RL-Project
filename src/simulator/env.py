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
import cProfile

    
# IMPORT DYNAMICS, CONTROL and USER CODE
import quad_dynamics as qd
import control
import tello
import visualizer

# Force reload custom modules and run the latest code
importlib.reload(control)
importlib.reload(qd)
importlib.reload(tello)
importlib.reload(visualizer)

class Env():
    def __init__(self):
        # CONSTANTS
        self.fps = 20
        self.SAFE_MAX = 50.
        self.SAFE_MAX_HEIGHT = 50.

        # STOP time for simulation
        self.sim_stop_time = 15

        # INIT RENDERING AND CONTROL
        self.controller = control.quad_control()

        # SET TIME STEP
        # env step time should be a multiple of control_dt
        self.env_step_dt = 0.02
        self.dynamics_dt = 0.005
        self.control_dt = self.controller.dt

        # Step the rate controller N times till time >= env_step_time
        # Step the plant M times for every control action
        self.N = round(self.env_step_dt/self.control_dt)
        self.M = round(self.control_dt/self.dynamics_dt)
        
        # reset for good measure
        self.observations = self.reset()    
    
    def reset(self, randomizeWPs = True):
        self.done = False
        self.timeExceeded = False
        self.hitGround = False
        self.heightTooLarge = False
        self.maxStateReached = False
        self.missionComplete = True

        # INIT STATES
        self.current_time = 0.
        xyz = np.array([0.0, 0.0, -5.0])
        vxyz = np.array([0.0, 0.0, 0.0])
        quat = np.array([1.0, .0, .0, .0])
        pqr = np.array([0.0, .0, .0])
        self.current_ned_state = np.concatenate((xyz, vxyz, quat, pqr))

        # INIT VARS
        self.current_time = 0.
        self.prevDist = 0.
        self.prevU = [0., 0., 0., 0.]

        # INIT WAYPOINTS
        # TODO implement randomization
        # self.waypoints_ned = np.array([[0, 0, -7.0],
        #                           [2, 2, -7.0],
        #                           [2, -2, -7.0],
        #                           [-2, -2, -7.0],
        #                           [-2, 2, -7.0],
        #                          ])
        self.waypoints_ned = np.array([[0, 0, -7.0]
                                 ])
        self.curr_wp_idx = 0

        # INIT LOG
        self.stateArray = self.current_ned_state
        self.timeArray = 0
        self.controlArray = np.array([0., 0, 0, 0])
        self.rewardArray = 0
        self.activeWaypointsArray = np.array([0., 0, 0])

        observations = np.zeros(17)
        observations[0:10] = self.current_ned_state[0:10]
        observations[10:13] = self.waypoints_ned[0]
        observations[13:17] = np.zeros(4) # RPYT (not rotor velocities, 1 future waypoint)
        return observations.astype(np.float32).reshape((17,))

    def step(self, u):
        # u = [x_rate, y_rate, z_rate, throttle] list
        # Rates are between -1 and 1
        # Throttle is between -1 and 1. 
        #   -1 means 0 thrust and 1 means max (it should hover around 0)
        # Probably a good idea to intialize network in a manner that it outputs around 0 during the start (so that the drone hovers without crashing)

        # dynamics is our common clock  

        if self.done == True:
            # early return
            return self.observations, 0, self.done, self.current_time
        
        # Step the rate controller N times till time >= env_step_time
        for i in range(1, self.N+1):
            pqr_sp = u[0:3]
            throttle = (u[3]+1)/2.0

            U = self.controller.step(self.current_ned_state, pqr_sp, throttle)
            # Step the plant M times
            for j in range(1, self.M+1):
                self.current_ned_state = self.current_ned_state + self.dynamics_dt*qd.model_derivative(self.current_time,
                                                                self.current_ned_state,
                                                                U,
                                                                tello)
                self.current_time += self.dynamics_dt
   
        # check if waypoint is complete or switch over
        xyz = self.current_ned_state[0:3]
        curr_wp = self.waypoints_ned[self.curr_wp_idx]
        dist = np.linalg.norm(curr_wp - xyz)

        reward_mission = 0.
        reward_wp = 0.

        reward_prog = (self.prevDist - dist)
        self.prevDist = dist

        # if dist<0.2:
        #     # waypoint complete
        #     reward_wp = 1.
        #     self.curr_wp_idx+=1
        #     if self.curr_wp_idx >= len(self.waypoints_ned):
        #         self.missionComplete = True
        #         self.done = True
        #         reward_mission = 1.
        #     else:
        #         # prevent sudden jump in prevDist when wp changes!
        #         curr_wp = self.waypoints_ned[self.curr_wp_idx]
        #         dist = np.linalg.norm(curr_wp - xyz)
        #         self.prevDist = dist

        # POPULATE OBSERVATIONS
            # first 10 are position (-20.0 to 20.0), velocity (-20.0 to 20.0), quaternion (-1 to 1) 10
            # current waypoint xyz (-20.0 to 20.0)) is an additional observation 3
            # previous actuator (-1 to 1) command 4
        observations = np.zeros(17)
        observations[0:10] = self.current_ned_state[0:10]
        observations[10:13] = curr_wp
        observations[13:17] = u # RPYT (not rotor velocities, 1 future waypoint)

        self.observations = observations.astype(np.float32).reshape((17,))
        # TERMINATION CONDITIONS
        if self.current_time>self.sim_stop_time:
            self.done = True
            self.timeExceeded = True
            # if hover: 
            #     self.missionComplete = True

        # Check height
        height = -self.current_ned_state[2]
        reward_crash = 0
        if height<0:
            self.done = True
            self.hitGround = True
            reward_crash = 1
        if height>self.SAFE_MAX_HEIGHT:
            self.done = True
            self.heightTooLarge = True
            reward_crash = 1
        
        if np.max(self.current_ned_state)>self.SAFE_MAX or np.min(self.current_ned_state)<-self.SAFE_MAX:
            self.done = True
            self.maxStateReached = True
            reward_crash = 1
        
        reward_actuator = 0.5*np.linalg.norm(u) + np.linalg.norm(u - self.prevU)
        self.prevU = u

        # REWARD FUNCTION
        reward = 100*reward_mission + 10*reward_wp + 3*reward_prog -0.5*reward_actuator -50*reward_crash

        # Log the signals
        self.stateArray = np.vstack((self.stateArray, self.current_ned_state))
        self.controlArray = np.vstack((self.controlArray, U))
        self.timeArray = np.append(self.timeArray, self.current_time)
        self.rewardArray = np.append(self.rewardArray, reward)
        self.activeWaypointsArray = np.vstack((self.activeWaypointsArray, self.waypoints_ned[self.curr_wp_idx]))

        return self.observations, reward, self.done, self.current_time

    def log(self, file):
        # Externion will automagically be added. please give only file name with relative path

        # SAVE LOGGED SIGNALS TO MAT FILE FOR POST PROCESSING IN MATLAB
        # loggedDict = {'time': self.timeArray,
        #             'state': self.stateArray,
        #             'control': self.controlArray,
        #             }
        # scipy.io.savemat('./simulator/log/states.mat', loggedDict)

        # import pdb
        # pdb.set_trace()

        np.savez(file + '_runtime', time=self.timeArray, state=self.stateArray, control=self.controlArray, reward=self.rewardArray, activeWP=self.activeWaypointsArray)
        np.savez(file + '_config', wps=self.waypoints_ned, env_step_dt=self.env_step_dt)

    def animate(self, file):
        # pass file name with extension
        visualizer.plot3d(file, self.waypoints_ned, self.stateArray, self.env_step_dt)
