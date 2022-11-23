import gym
from gym import spaces
import numpy as np
import random
from pump_sim_dis import hybrid_pump
from pump_env import PumpEnv


P_0 = 1.01*1e5  # Pa
MAX_STEP = 100
CHAMBER_LEN = 0.1

# This is a variant of the basic env

class PumpEnvVar(PumpEnv):
    def __init__(self, var_L_range=[0.0,0.02], goal_pressure_range=[1.1, 4.0]):
        # super(PumpEnv, self).__init__()
        # Continuous actions
        self.action_space = spaces.Box(low=-1, high=1,
                                            shape=(2,), dtype=np.float32)
        # Input observation:
        dim_obs = 10
        n_stack_obs = 2
        dim_stack_obs = int(dim_obs * n_stack_obs)
        self.observation_space = spaces.Box(low=-1.0, high=1.0,
                                            shape=(dim_stack_obs,), dtype=np.float32)
        self.var_L_range = var_L_range
        self.pump = hybrid_pump(L_L=0.1, L_R=0.1+random.uniform(self.var_L_range[0],self.var_L_range[1]))
        self.action_counter = 0
        self.reward = 0
        self.goal_pressure_range = goal_pressure_range
        self.goal_pressure = 0
        self.prev_observation = np.zeros((dim_obs,))
        self.sequence_num = 0
        

    def reset(self):
        ## Initialization
        self.done = False
        self.reward = 0
        self.prev_shaping = None
        self.action_counter = 0

        # Pump initialization with random load
        self.pump = hybrid_pump(L_L=CHAMBER_LEN, L_R=CHAMBER_LEN+random.uniform(self.var_L_range[0],self.var_L_range[1]))  # train with random load
        
        # Set random goal pressure
        self.goal_pressure = random.uniform(self.goal_pressure_range[0], self.goal_pressure_range[1]) * P_0

        # Calculate observation
        self.step_observation = np.array([self.goal_pressure, float(self.goal_pressure < P_0),
                                self.pump.Lchamber.P, self.pump.Rchamber.P,
                                CHAMBER_LEN + self.pump.P_M, CHAMBER_LEN - self.pump.P_M,  # chamber length
                                self.pump.P_M,
                                self.pump.valve[0], self.pump.valve[1], self.pump.valve[2],
                                ])
        self.step_observation = self.normalization(self.step_observation)
        observation = np.concatenate((self.step_observation, self.step_observation)) # stack two observations

        return observation
    

    def set_load_and_goal_pressure(self, var_L, goal_pressure):
        # Specify load and goal for testing
        self.pump = hybrid_pump(L_L=CHAMBER_LEN, L_R=CHAMBER_LEN+var_L)
        self.goal_pressure = goal_pressure
        self.step_observation = np.array([self.goal_pressure, float(self.goal_pressure < P_0),
                                self.pump.Lchamber.P, self.pump.Rchamber.P, 
                                CHAMBER_LEN + self.pump.P_M, CHAMBER_LEN - self.pump.P_M,  # chamber length
                                self.pump.P_M,
                                self.pump.valve[0], self.pump.valve[1], self.pump.valve[2],
                                ])
        self.step_observation = self.normalization(self.step_observation)
        observation = np.concatenate((self.step_observation, self.step_observation))
        return observation

    
if "__main__" == __name__:
    CM_2_M = 0.01
    env = PumpEnvVar(var_L=0.02)
    env.reset()
    print(env.pump.Rchamber.V)
    env.render()
    
    action=[+1.0, -1.0]
    env.step(action)
    env.render()

    # for i in range(100):
    #     action = env.action_space.sample()
    #     env.step(action)
    #     print('action:', action)
    #     env.render(0)