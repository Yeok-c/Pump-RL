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
                                            shape=(2,), dtype=np.float32)
        # Input observation:
        dim_obs = 10
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



    def step(self, action):
        # Execute one time step within the environment
        self.action_counter += 1

        # Reset valve for visualization
        self.pump.close_R_valve()
        self.pump.close_L_valve()
        self.pump.open_inner_valve()

        # [Action 0]: P_M
        prev_P_M = self.pump.P_M
        goal_P_M = action[0] * 0.05
        move_distance = goal_P_M - prev_P_M
        if move_distance >= 0:
            self.pump.move_motor_to_R(move_distance)
        elif move_distance < 0:
            self.pump.move_motor_to_L(-move_distance)
        else:
            print("Action[0] error")
        
        #[Action 1]: Valve  (Change P_M first)
        if action[1] > 0.5 and action[1] <= 1:
            self.pump.open_R_valve()
        elif action[1] > 0:
            self.pump.open_inner_valve()
        elif action[1] > -0.5:
            self.pump.open_L_valve()
        elif action[1] >= -1:
            pass
        else:
            print("Action[1] error")
        
        # Check if the pump is done
        info = {}
        done_threshold = 0.01
        if abs(self.pump.Rchamber.P - self.goal_pressure)/self.goal_pressure < done_threshold:
            self.done = True
            self.reward = 0.0
        elif self.action_counter > MAX_STEP:
            self.done = True
            self.reward = -1.0
            info["episode_timeout"] = True
        else:
            self.done = False
            self.reward = -1.0
        
        # Calculate observation
        prev_observation = self.step_observation
        self.step_observation = np.array([self.goal_pressure, float(self.goal_pressure < P_0),
                                            self.pump.Lchamber.P, self.pump.Rchamber.P, 
                                            # self.pump.Lchamber.V, self.pump.Rchamber.V,  # V is not available in the real pump
                                            CHAMBER_LEN + self.pump.P_M, CHAMBER_LEN - self.pump.P_M,
                                            self.pump.P_M,
                                            self.pump.valve[0], self.pump.valve[1], self.pump.valve[2],  # converge faster but not stable
                                            # self.load  # load
                                            ])
        # Normalize observation
        self.step_observation = self.normalization(self.step_observation)
        # Stack 2 frames (can make convergence faster)
        observation = np.concatenate((prev_observation, self.step_observation))
        
        return observation, self.reward, self.done, info

    def reset(self):
        ## Initialization
        self.done = False
        self.reward = 0
        self.prev_shaping = None
        self.action_counter = 0

        # Pump initialization with random load
        # self.load = random.uniform(self.load_range[0],self.load_range[1])
        # self.pump = hybrid_pump(L_L=CHAMBER_LEN, L_R=CHAMBER_LEN, load=self.load)  # train with random load
        
        # Set random goal pressure
        self.goal_pressure = random.uniform(self.goal_pressure_range[0], self.goal_pressure_range[1]) * P_0

        # Calculate observation
        self.step_observation = np.array([self.goal_pressure, float(self.goal_pressure < P_0),
                                            self.pump.Lchamber.P, self.pump.Rchamber.P,
                                            CHAMBER_LEN + self.pump.P_M, CHAMBER_LEN - self.pump.P_M,  # chamber length
                                            self.pump.P_M,
                                            self.pump.valve[0], self.pump.valve[1], self.pump.valve[2],
                                            # self.load  # load
                                            ])
        self.step_observation = self.normalization(self.step_observation)
        try:
            observation = np.concatenate((self.prev_observation, self.step_observation)) # stack two observations
        
        except:
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
                                # self.load  # load scale
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


    # env.set_load_and_goal_pressure(0.01, 1.1*P_0)
    # for i in range(5):
    #     action = env.action_space.sample()
    #     obs, _, _, _ = env.step(action)
    #     print('obs:', obs[-1], env.var_L)
    #     env.render(0)

    # Sequence 3: Positive and negative pressure with 3 chambers
    action_names = ([
        "",
        "",
        "",
        "",
        "",
        "",
    ])
    
    action_sequence = np.array([
        # [ 1, 0, 0, 0, 1, 0], # (a) Max right position, close right valve
        # [ 1, 0, 0, 1, 0, 0], # (a) Max right position, open right valve
        # [ 1, 0, 0, 0, 0, 0], # (b) Max right position, close all valve
        # [-1, 0, 1, 0, 0, 0], # (c) Max left position, open bottom valve
        # [-1, 0, 0, 1, 0, 0], # (d) Max left position, open right valve
        # [-1, 0, 0, 0, 1, 0], # "(d) Max left position, open load valve",
        [ 1, 0.3], # (a) Max right position, close right valve
        [-1, 0.3], # (a) Max right position, open right valve
        [ 1, 0.3], # (b) Max right position, close all valve
        [-1, 0.3], # (c) Max left position, open bottom valve
        [ 1, 0.3], # (d) Max left position, open right valve
        [-1, 0.3], # "(d) Max left position, open load valve",
        
        ])

    while 1:
        for name, action in zip(action_names, action_sequence):
            # action = env.action_space.sample()
            obs, _, _, _ = env.step(action)
            print(name, action)
            env.render(0, title=name)