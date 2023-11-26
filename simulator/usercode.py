import numpy as np
import cv2
import bpy
import os
import scipy

class state_machine:
    def __init__(self):
        # User code executes every dt time in seconds
        self.dt = 0.050

        # Logger
        self.time_array = 0
        self.x_sp_array = 0
        self.y_sp_array = 0
        self.z_sp_array = 0

        self.x_array = 0
        self.y_array = 0
        self.z_array = 0

        self.vx_array = 0
        self.vy_array = 0
        self.vz_array = 0
        self.vx_sp_array = 0
        self.vy_sp_array = 0
        self.vz_sp_array = 0

    def step(self, time, currpos, currvel):        
        # FOLLOW GENERATED TRAJECTORY
        xyz_desired = self.MP[:3, self.activeIndex]

        if self.activeIndex<len(self.MP[1, :]):
            self.activeIndex = self.activeIndex+1
        
        # logger
        self.time_array = np.vstack((self.time_array, time))
        self.x_array = np.vstack((self.x_array, currpos[0]))
        self.y_array = np.vstack((self.y_array, currpos[1]))
        self.z_array = np.vstack((self.z_array, currpos[2]))
        self.x_sp_array = np.vstack((self.x_sp_array, xyz_desired[0]))
        self.y_sp_array = np.vstack((self.y_sp_array, xyz_desired[1]))
        self.z_sp_array = np.vstack((self.z_sp_array, xyz_desired[2]))

        self.vx_array = np.vstack((self.vx_array, currvel[0]))
        self.vy_array = np.vstack((self.vy_array, currvel[1]))
        self.vz_array = np.vstack((self.vz_array, currvel[2]))
        self.vx_sp_array = np.vstack((self.vx_sp_array, vel_desired[0]))
        self.vy_sp_array = np.vstack((self.vy_sp_array, vel_desired[1]))
        self.vz_sp_array = np.vstack((self.vz_sp_array, vel_desired[2]))

        return xyz_desired, vel_desired, acc_desired, yaw_setpoint

    def terminate(self):
        loggedDict = {'time': self.time_array,
                  'x': self.x_array,
                  'y': self.y_array,
                  'z': self.z_array,
                  'x_des': self.x_sp_array,
                  'y_des': self.y_sp_array,
                  'z_des': self.z_sp_array,
                  'vx': self.vx_array,
                  'vy': self.vy_array,
                  'vz': self.vz_array,
                  'vx_des': self.vx_sp_array,
                  'vy_des': self.vy_sp_array,
                  'vz_des': self.vz_sp_array,
                  }  
        scipy.io.savemat('./log/user_states.mat', loggedDict)
        print('user state machine terminted')


    def fetchLatestImage(self):
        # Fetch image - renders the camera, saves the rendered image to a file and reads from it. 
        path_dir = bpy.data.scenes["Scene"].node_tree.nodes["File Output"].base_path

        # Render Drone Camera
        cam = bpy.data.objects['DownCam']    
        bpy.context.scene.camera = cam
        bpy.context.scene.render.filepath = os.path.join(path_dir, 'DownCam_latest.png')
        bpy.ops.render.render(write_still=True)

        return cv2.imread(bpy.context.scene.render.filepath)
    
