import gym
from gym import spaces
import numpy as np
import random
# from pump_sim_dis import hybrid_pump
from pump_sim_load import hybrid_pump
from pump_env import PumpEnv


P_0 = 1.01*1e5  # Pa
MAX_STEP = 100
CHAMBER_LEN = 0.1

# This is a variant of the basic env

class PumpEnvVar(PumpEnv):
    def __init__(self, load_range=[0.0,2.0], goal_pressure_range=[1.1, 2.0]):
        # super(PumpEnv, self).__init__()
        # Continuous actions
        self.action_space = spaces.Box(low=-1, high=1,
                                            shape=(4,), dtype=np.float32)
        # Input observation:
        dim_obs = 11
        n_stack_obs = 2
        dim_stack_obs = int(dim_obs * n_stack_obs)
        self.observation_space = spaces.Box(low=-1.0, high=1.0,
                                            shape=(dim_stack_obs,), dtype=np.float32)
        self.load_range = load_range
        self.load = random.uniform(self.load_range[0],self.load_range[1])
        self.pump = hybrid_pump(L_L=CHAMBER_LEN, L_R=CHAMBER_LEN, load=self.load)
        self.action_counter = 0
        self.reward = 0
        self.goal_pressure_range = goal_pressure_range
        self.goal_pressure = 0
        self.prev_observation = np.zeros((dim_obs,))


    def reset(self):
        ## Initialization
        self.done = False
        self.reward = 0
        self.prev_shaping = None
        self.action_counter = 0

        # Pump initialization with random load
        self.load = random.uniform(self.load_range[0],self.load_range[1])
        self.pump = hybrid_pump(L_L=CHAMBER_LEN, L_R=CHAMBER_LEN, load=self.load)  # train with random load
        
        # Set random goal pressure
        self.goal_pressure = random.uniform(self.goal_pressure_range[0], self.goal_pressure_range[1]) * P_0

        # Calculate observation
        self.step_observation = np.array([self.goal_pressure, float(self.goal_pressure < P_0),
                                            self.pump.Lchamber.P, self.pump.Rchamber.P,
                                            CHAMBER_LEN + self.pump.P_M, CHAMBER_LEN - self.pump.P_M,  # chamber length
                                            self.pump.P_M,
                                            self.pump.valve[0], self.pump.valve[1], self.pump.valve[2],
                                            self.load  # load
                                            ])
        self.step_observation = self.normalization(self.step_observation)
        observation = np.concatenate((self.step_observation, self.step_observation)) # stack two observations

        return observation
    

    def set_load_and_goal_pressure(self, load, goal_pressure):
        # Specify load and goal for testing
        self.load = load
        self.pump = hybrid_pump(L_L=CHAMBER_LEN, L_R=CHAMBER_LEN, load=load)
        self.goal_pressure = goal_pressure
        self.step_observation = np.array([self.goal_pressure, float(self.goal_pressure < P_0),
                                self.pump.Lchamber.P, self.pump.Rchamber.P, 
                                CHAMBER_LEN + self.pump.P_M, CHAMBER_LEN - self.pump.P_M,  # chamber length
                                self.pump.P_M,
                                self.pump.valve[0], self.pump.valve[1], self.pump.valve[2],
                                self.load  # load scale
                                ])
        self.step_observation = self.normalization(self.step_observation)
        observation = np.concatenate((self.step_observation, self.step_observation))
        return observation

    
if "__main__" == __name__:
    env = PumpEnvVar()
    obs = env.reset()
    print('obs', obs)
    # print(env.var_L)
    # env.render()
    
    # action=[+1.0, -1.0]
    # obs, reward, done, info = env.step(action)
    # print('obs', obs)
    # env.render()
    # obs = env.reset()
    # print('obs', obs)

    for i in range(5):
        action = env.action_space.sample()
        obs, _, _, _ = env.step(action)
        print('obs:', obs[-1], env.var_L)
        env.render(0)

    env.set_load_and_goal_pressure(0.01, 1.1*P_0)
    for i in range(5):
        action = env.action_space.sample()
        obs, _, _, _ = env.step(action)
        print('obs:', obs[-1], env.var_L)
        env.render(0)