import sys
sys.path.append('./simulator')

from env import Env
import numpy as np
import time
# import cProfile

def main():
    # The environment is simple threadsafe python object
    env = Env()
    env.reset()
    np.set_printoptions(suppress=True, precision=3, linewidth=200)

    done = False
    startTime = time.time()

    while not done:
        u = np.array([0.00, 0.0, 0.0, 0.0])
        observations, reward, done, simtime = env.step(u)

        print(observations)
    print(time.time() - startTime)


# cProfile.run('main()')
main()