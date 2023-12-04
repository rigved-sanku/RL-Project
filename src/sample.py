import sys
sys.path.append('./simulator')

# How to run this file. From the folder containing sample.py, run python sample.py in terminal. install the required depenedencies such as scipy numpy pyquaternion

from env import Env
import numpy as np
import time
import visualizer

def main():
    # The environment is simple threadsafe python object
    env = Env()
    env.reset()
    np.set_printoptions(suppress=True, precision=3, linewidth=200)

    done = False
    startTime = time.time()

    while not done:
        u = np.array([0.01, 0.0, 0.0, 1])
        observations, reward, done, simtime = env.step(u)
        print(observations)
        # print(reward)
        # print(done)
    
    # log the current states of env 
    env.log('testlog')
    # animate right now and store into testvideo.mp4
    env.animate('testvideo.mp4')

    # animate from the log. Can be called outside
    visualizer.plot3dFromFile('testvideoFromLog.mp4', 'testlog')


    print(time.time() - startTime)


main()