import gym
from gym import spaces
import numpy as np
import random
# from pump_sim_dis import hybrid_pump
# from pump_sim_load import hybrid_pump
from pump_sim_load_addvalve import hybrid_pump
from pump_env import PumpEnv


P_0 = 1.01*1e5  # Pa
MAX_STEP = 150
CHAMBER_LEN = 0.1

# This is a variant of the basic env

class PumpEnvVar_Two(PumpEnv):
    def __init__(self, 
    load_range=[0.0,2.0], 
    goal_pressure_L_range=[0.7, 0.9],
    goal_pressure_R_range=[1.1, 2.0],
    max_episodes = MAX_STEP,
    use_combined_loss = False,
    use_step_loss = True
    ):
        # super(PumpEnv, self).__init__()
        # Continuous actions
        self.action_space = spaces.Box(low=-1, high=1,
                                            shape=(6,), dtype=np.float32)
        # Input observation:
        dim_obs = 12
        n_stack_obs = 2
        dim_stack_obs = int(dim_obs * n_stack_obs)
        self.observation_space = spaces.Box(low=-1.0, high=1.0,
                                            shape=(dim_stack_obs,), dtype=np.float32)
        self.load_range = load_range
        self.load_L = random.uniform(self.load_range[0],self.load_range[1])
        self.load_R = random.uniform(self.load_range[0],self.load_range[1])
        self.pump = hybrid_pump(
            L_L=CHAMBER_LEN, L_R=CHAMBER_LEN, 
            load_chamber_ratio_L=self.load_L, 
            load_chamber_ratio_R=self.load_R,
        )
        self.action_counter = 0
        self.reward = 0
        self.goal_pressure_L_range = goal_pressure_L_range
        self.goal_pressure_R_range = goal_pressure_R_range
        self.goal_pressure_L = 0
        self.goal_pressure_R = 0
        self.prev_observation = np.zeros((dim_obs,))
        self.max_episodes = max_episodes
        self.use_combined_loss = use_combined_loss
        self.use_step_loss = use_step_loss

    def step(self, action):
        # Execute one time step within the environment
        self.action_counter += 1

        # Reset valve for visualization
        self.pump.close_R_valve()
        self.pump.close_L_valve()
        self.pump.close_inner_valve()
        self.pump.close_L_load_valve()
        self.pump.close_R_load_valve()

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
        # if action[1] > 0.5 and action[1] <= 1:
        #     self.pump.open_R_valve()
        # elif action[1] > 0:
        #     self.pump.open_inner_valve()
        # elif action[1] > -0.5:
        #     self.pump.open_L_valve()
        # elif action[1] >= -1:
        #     pass
        # else:
        #     print("Action[1] error")
        
        pump_action = np.argmax(action[1:])
        if pump_action == 0:
            self.pump.open_R_valve()
        if pump_action == 1:
            self.pump.open_inner_valve()
        if pump_action == 2:
            self.pump.open_L_valve()
        if pump_action == 3:
            self.pump.open_L_load_valve()
        if pump_action == 4:
            self.pump.open_R_load_valve()

        # if action[1] > 0:
        #     self.pump.open_L_valve()
        # if action[2] > 0:
        #     self.pump.open_inner_valve()
        # if action[3] > 0:
        #     self.pump.open_R_valve()
        # if action[4] > 0:
        #     self.pump.open_L_load_valve()
        # if action[5] > 0:
        #     self.pump.open_R_load_valve()

        # Loss functions
        loss_L = abs(self.pump.Lchamber.P - self.goal_pressure_L)/P_0 
        loss_R = abs(self.pump.Rchamber.P - self.goal_pressure_R)/P_0
        # loss_L = abs(self.pump.Lchamber.P - self.goal_pressure_L)/self.goal_pressure_L 
        # loss_R = abs(self.pump.Rchamber.P - self.goal_pressure_R)/self.goal_pressure_R
        combined_loss = loss_L + loss_R

        # Check if the pump is done
        info = {}
        done_threshold = 0.01
        
        # if achieved goals already, leave
        if (loss_L < done_threshold) and (loss_R < done_threshold):
            self.done = True
            self.reward = 0.0

        # if exceed maximum episodes, leave
        elif self.action_counter > self.max_episodes:
            self.done = True
            info["episode_timeout"] = True
        
        # if still not done
        else:
            self.done = False
            if self.use_step_loss == True:
                self.reward = -1.0
            
            if self.use_combined_loss == True:
                self.reward -= combined_loss          
        
        


        prev_observation = self.step_observation
        self.step_observation = self.calculate_step_observations()
    
            
        # Stack 2 frames (can make convergence faster)
        observation = np.concatenate((prev_observation, self.step_observation))
        
        return observation, self.reward, self.done, info

    def calculate_step_observations(self):
        # Calculate observation
        self.step_observation = np.array([
            self.goal_pressure_L, float(self.goal_pressure_L < P_0),
            self.goal_pressure_R, float(self.goal_pressure_R < P_0),
            self.pump.Lchamber.P, self.pump.Rchamber.P, 
            # self.pump.Lchamber.V, self.pump.Rchamber.V,  # V is not available in the real pump
            CHAMBER_LEN + self.pump.P_M, CHAMBER_LEN - self.pump.P_M,
            self.pump.P_M,
            self.pump.valve[0], self.pump.valve[1], self.pump.valve[2],  # converge faster but not stable
            # self.load  # load
            ])
        # Normalize observation
        step_observation = self.normalization(self.step_observation)
        return step_observation 


    def normalization(self, observation):
        # Normalize observation
        goal_pressure_min = 0.05 * P_0
        goal_pressure_max = 10 * P_0
        val_L_min = 0.0
        val_L_max = 2.0
        float_offset=0.001
        observation_range_low = np.array([goal_pressure_min, 0.0, goal_pressure_min, 0.0, 0.01*P_0, 0.01*P_0, 0.049, 0.049, -0.05, 0.0, 0.0, 0.0])
        observation_range_high= np.array([goal_pressure_max, 1.0, goal_pressure_max, 1.0, 10*P_0, 10*P_0, 0.151, 0.151, +0.05, 1.0, 1.0, 1.0])

        observation_range_low = observation_range_low - float_offset
        observation_range_high = observation_range_high + float_offset

        norm_h, norm_l = 1.0, -1.0
        norm_observation = (observation - observation_range_low) / (observation_range_high - observation_range_low) * (norm_h - norm_l) + norm_l
        if ((norm_l <= norm_observation) & (norm_observation <= norm_h)).all():
            pass
        else:
            print("obs:", observation)
            print("norm_obs:", norm_observation)
            # raise Exception("Normalization error")
        return norm_observation


    def reset(self):
        ## Initialization
        self.done = False
        self.reward = 0
        self.prev_shaping = None
        self.action_counter = 0

        # Pump initialization with random load
        self.load_L = random.uniform(self.load_range[0],self.load_range[1])
        self.load_R = random.uniform(self.load_range[0],self.load_range[1])
        self.pump = hybrid_pump(
            L_L=CHAMBER_LEN, L_R=CHAMBER_LEN, 
            load_chamber_ratio_L=self.load_L, 
            load_chamber_ratio_R=self.load_R,
            )  # train with random load
        
        # Set random goal pressure
        self.goal_pressure_L = random.uniform(self.goal_pressure_L_range[0], self.goal_pressure_L_range[1]) * P_0
        self.goal_pressure_R = random.uniform(self.goal_pressure_R_range[0], self.goal_pressure_R_range[1]) * P_0

        # Calculate observation
        self.step_observation = self.calculate_step_observations()
        
        try:
            observation = np.concatenate((self.prev_observation, self.step_observation)) # stack two observations
        
        except:
            observation = np.concatenate((self.step_observation, self.step_observation)) # stack two observations

        return observation
    

    def set_load_and_goal_pressure(self, load_L, load_R, goal_pressure_L, goal_pressure_R):
        # Specify load and goal for testing
        self.load_L = load_L
        self.load_R = load_R
        self.pump = hybrid_pump(
            L_L=CHAMBER_LEN, L_R=CHAMBER_LEN, 
            load_chamber_ratio_L=self.load_L, 
            load_chamber_ratio_R=self.load_R,
            )  # train with random load
        
        self.goal_pressure_L = goal_pressure_L
        self.goal_pressure_R = goal_pressure_R
        
        self.step_observation = self.calculate_step_observations()
        observation = np.concatenate((self.step_observation, self.step_observation))
        return observation

    def print_step_info(self):
        print("Lchamber.P: {p1:.3f} Rchamber.P: {p2:.3f}| ".format(p1=self.pump.Lchamber.P/P_0, p2=self.pump.Rchamber.P/P_0),
              "Goal L {p1:.3f} | Goal R {p2:.3f} ".format(p1=self.goal_pressure_L/P_0, p2=self.goal_pressure_R/P_0),
            #   "Goal {p1} pressure {p2:.3f} | ".format(p1=self.goal_sequence_number, p2=self.goal_pressure/P_0),
              "Step reward: {p} | ".format(p=self.reward),
              "Error L: {p1:.3f} | Error R: {p2:.3f}".format(
                p1 = (self.pump.Lchamber.P - self.goal_pressure_L)/self.goal_pressure_L,
                p2 = (self.pump.Rchamber.P - self.goal_pressure_R)/self.goal_pressure_R
                )
        )
if "__main__" == __name__:
    env = PumpEnvVar_Two()
    obs = env.reset()
    print('obs', obs)
    env.render(0, title='Reset')
    # # Sequence 1: same as paper
    # action_names = ([
    #     '(a,b) Max left pos, open left valve',
    #     '(c) Max left pos, close all valve',
    #     '(d) Max left pos, open bottom valve',
    #     '(e) Max left pos, close bottom valve',
    #     '(f) Max left pos, open right valve',
    #     '(g) Max right pos, open right valve',
    #     '(h) Max right pos, close all valve',
    #     '(i) Max right pos, open bottom valve',
    #     '(j) Max right pos, close all valve',
    #     '(k) Max right pos, open left valve',])
    # action_sequence = 0.5*np.array([
    #     [-1, 1, 0, 0], # (a,b) Max left position, open left valve
    #     [-1, 0, 0, 0], # (c) Close all valve
    #     [-1, 0, 1, 0], # (d) open bottom valve
    #     [-1, 0, 0, 0], # (e) Close bottom valve
    #     [-1, 0, 0, 1], # (f) Open right valve
    #     [ 1, 0, 0, 1], # (g) Max right position while opening right valve
    #     [ 1, 0, 0, 0], # (h) Max right position while close all valve
    #     [ 1, 0, 1, 0], # (i) Max right position while opening bottom valve
    #     [ 1, 0, 0, 0], # (j) Max right position while close all valve
    #     [ 1, 1, 0, 0], # (k) Max right position while open left valve
    #     ])

    # # Sequence 2: positive pressure only as seen before
    # action_names = ([
    #     '(a) Max right position, open left valve',
    #     '(b) Max right position, close all valve',
    #     '(c) Max left position, open bottom valve',
    #     '(d) Max left position, open left valve',
    # ])
    
    # action_sequence = np.array([
    #     [ 1, 1, 0, 0], # (a) Max right position, open left valve
    #     [ 1, 0, 0, 0], # (b) Max right position, close all valve
    #     [-1, 0, 1, 0], # (c) Max left position, open bottom valve
    #     [-1, 1, 0, 0], # (d) Max left position, open left valve
    #     ])
    
    # Sequence 3: negative pressure (reverse of what we seen before)
    # action_names = ([
    #     '(a) Max right position, open right valve',
    #     '(b) Max right position, close all valve',
    #     '(c) Max left position, open bottom valve',
    #     '(d) Max left position, open right valve',
    # ])
    
    # action_sequence = np.array([
    #     [ 1, 0, 0, 1], # (a) Max right position, open right valve
    #     [ 1, 0, 0, 0], # (b) Max right position, close all valve
    #     [-1, 0, 1, 0], # (c) Max left position, open bottom valve
    #     [-1, 0, 0, 1], # (d) Max left position, open right valve
    #     ])
    
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
        [ 1, 0, 1, 0, 0, 0], # (a) Max right position, close right valve
        [-1, 0, 1, 0, 0, 0], # (a) Max right position, open right valve
        [ 1, 0, 1, 0, 0, 0], # (b) Max right position, close all valve
        [-1, 0, 1, 0, 0, 0], # (c) Max left position, open bottom valve
        [ 1, 0, 1, 0, 0, 0], # (d) Max left position, open right valve
        [-1, 0, 1, 0, 0, 0], # "(d) Max left position, open load valve",
        
        ])

    while 1:
        for name, action in zip(action_names, action_sequence):
            # action = env.action_space.sample()
            obs, _, _, _ = env.step(action)
            print(name, action)
            env.render(0, title=name)

    # for i in range(env.max_episodes):
    #     action = env.action_space.sample()
    #     obs, _, _, _ = env.step(action)
    #     # print('obs:', obs[-1], env.var_L)
    #     env.print_step_info()
    #     env.render(0)

    # env.set_load_and_goal_pressure(0.01, 1.1*P_0)
    # for i in range(5):
    #     action = env.action_space.sample()
    #     obs, _, _, _ = env.step(action)
    #     # print('obs:', obs[-1], env.var_L)
    #     env.print_step_info() 
    #     env.render(0)