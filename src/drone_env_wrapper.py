from simulator.env import Env
import tf_agents 
import numpy as np

class DroneEnv(tf_agents.environments.py_environment.PyEnvironment):

    def __init__(self, discount=1.0):
        super().__init__()
        self.discount = discount

        self._action_spec = tf_agents.specs.BoundedArraySpec(shape=(4,), 
                                                             dtype=np.float32, 
                                                             name="action", 
                                                             minimum=[-1.,-1.,-1.,-1], 
                                                             maximum=[1.,1.,1.,1])

        
        min_limits = np.array([-3.4028235e+38,-3.4028235e+38, -3.4028235e+38, \
            -3.4028235e+38, -3.4028235e+38, -3.4028235e+38, \
            -1.,-1.,-1.,-1., \
            -3.4028235e+38,-3.4028235e+38,-3.4028235e+38, \
            -1.,-1.,-1.,-1.])
        max_limits = -1*min_limits

        self._observation_spec = tf_agents.specs.BoundedArraySpec(shape=(17, ), 
                                                                  dtype=np.int32, 
                                                                  name="observation", 
                                                                  minimum=list(min_limits), 
                                                                  maximum=list(max_limits))

        self._drone_env = Env()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        obs = self._drone_env.reset()
        return tf_agents.trajectories.time_step.restart(obs)

    def _step(self, action):
        obs, reward, done, _ = self._drone_env.step(action)
        if done:
            return tf_agents.trajectories.time_step.termination(obs, reward)
        else:
            return tf_agents.trajectories.time_step.transition(obs, reward)
