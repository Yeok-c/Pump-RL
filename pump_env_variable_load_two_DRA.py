import gym
from gym import spaces
import numpy as np
import random
# from pump_sim_dis import hybrid_pump
# from pump_sim_load import hybrid_pump
from pump_sim_load_addvalve_dra import hybrid_pump
from pump_env import PumpEnv
import matplotlib.pyplot as plt
from scipy import signal
import time
import pickle

P_0 = 1.01*1e5  # Pa
MAX_STEP = 100
CHAMBER_LEN = 0.048
LOAD_V_RANGE = np.array([0, 0.1])




class PumpEnvVar_Two(PumpEnv):
    def __init__(self, 
    load_range=[0.0,2.0], 
    goal_pressure_L_range=[0.7, 0.9],
    goal_pressure_R_range=[1.1, 2.0],
    max_episodes = MAX_STEP,
    use_combined_loss = False,
    use_step_loss = True,
    obs_noise = 0.00,
    K_deform = 0.00,
    ):

        # super(PumpEnv, self).__init__()
        self.observation_name = [
        "Goal_L[t]", "Goal_L[t+1]", 
        # "Goal_L[t+2]", "Goal_R[t]", "Goal_R[t+1]", "Goal_R[t+2]",
        "self.pump.Lchamber.load_P", "self.pump.Rchamber.load_P", "self.pump.Lchamber.P", "self.pump.Rchamber.P", 
        # self.pump.Lchamber.V, self.pump.Rchamber.V,  # V is not available in the real pump
        "CHAMBER_LEN + self.pump.P_M", "CHAMBER_LEN - self.pump.P_M", "self.pump.P_M",
        "self.pump.valve[0]", "self.pump.valve[1]", "self.pump.valve[2]", "self.pump.valve[3]", "self.pump.valve[4]", 
        "self.pump.V_L", "self.pump.V_R"
        ]

        # Normalize observation
        goal_pressure_min = 0.05 * P_0
        goal_pressure_max = 10 * P_0
        val_L_min = 0.0
        val_L_max = 2.0
        self.float_offset=0.001

        self.observation_range_low = np.array([
            goal_pressure_min, goal_pressure_min, 
            goal_pressure_min, goal_pressure_min, 
            # goal_pressure_min, goal_pressure_min, 
            0.01*P_0, 0.01*P_0, 
            0.01*P_0, 0.01*P_0, 
            0.030, 0.030, -0.015, 
            0.0, 0.0, 0.0, 0.0, 0.0,
            LOAD_V_RANGE[0], LOAD_V_RANGE[0]
            # self.load_range[0], self.load_range[0]
        ])

        self.observation_range_high= np.array([
            goal_pressure_max, goal_pressure_max,
            goal_pressure_max, goal_pressure_max,
            # goal_pressure_max, goal_pressure_max,
            10*P_0, 10*P_0, 
            10*P_0, 10*P_0, 
            0.066, 0.066, +0.015, 
            1.0, 1.0, 1.0, 1.0, 1.0,
            LOAD_V_RANGE[1], LOAD_V_RANGE[1]
            # self.load_range[1], self.load_range[1]
        ])
        
        self.last_time = time.time()
        self.obs_noise = obs_noise
        self.K_deform = K_deform
        # Continuous actions
        self.action_space = spaces.Box(low=-1, high=1,
                                            shape=(7,), dtype=np.float32)
        # Input observation:
        self.future_frames = 2
        dim_obs = 14+self.future_frames*2
        n_stack_obs = 1
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
            K_deform=self.K_deform
        )

        self.action_counter = 0
        self.reward = 0

        self.max_episodes = max_episodes
        self.use_combined_loss = use_combined_loss
        self.use_step_loss = use_step_loss

        self.goal_pressure_L_range = goal_pressure_L_range
        self.goal_pressure_R_range = goal_pressure_R_range
        self.goal_pressure_sequence_L = np.zeros((self.max_episodes+self.future_frames,))
        self.goal_pressure_sequence_R = np.zeros((self.max_episodes+self.future_frames,))

        self.goal_pressure_L = self.goal_pressure_sequence_L[self.action_counter]
        self.goal_pressure_R = self.goal_pressure_sequence_R[self.action_counter]
        self.prev_observation = np.zeros((dim_obs,))
        self.reset()


    def step(self, action):
        self.last_time = time.time()
        # Execute one time step within the environment
        self.action_counter += 1
        self.reward = 0

        # Reset valve for visualization
        self.pump.close_R_valve()
        self.pump.close_L_valve()
        self.pump.close_inner_valve()
        self.pump.close_L_load_valve()
        self.pump.close_R_load_valve()

        # [Action 0]: P_M
        prev_P_M = self.pump.P_M
        goal_P_M = action[0]* 0.008 # * 0.05
        move_distance = goal_P_M - prev_P_M
        if move_distance >= 0:
            self.pump.move_motor_to_R(move_distance)
        elif move_distance < 0:
            self.pump.move_motor_to_L(-move_distance)
        else:
            print("Action[0] error")
        
        self.tim("Move motor")

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
            pass
        if pump_action == 1:
            self.pump.open_L_load_valve()
        if pump_action == 2:
            self.pump.open_L_valve()
        if pump_action == 3:
            self.pump.open_R_valve()
        if pump_action == 4:
            self.pump.open_R_load_valve()
        if pump_action == 5:
            self.pump.open_inner_valve()
        
        self.tim("Open valve")

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
        # self.loss_L = abs(self.pump.Lchamber.load_P - self.goal_pressure_L)/P_0 
        # self.loss_R = abs(self.pump.Rchamber.load_P - self.goal_pressure_R)/P_0
        # loss_L = abs(self.pump.Lchamber.P - self.goal_pressure_L)/self.goal_pressure_L 
        # loss_R = abs(self.pump.Rchamber.P - self.goal_pressure_R)/self.goal_pressure_R
        self.loss_L = np.power((self.pump.Lchamber.load_P - self.goal_pressure_L)/P_0, 2) 
        self.loss_R = np.power((self.pump.Rchamber.load_P - self.goal_pressure_R)/P_0, 2)
        combined_loss = np.sqrt(self.loss_L + self.loss_R)


        # Check if the pump is done
        info = {}
        # done_threshold = 0.01        
        # # if achieved goals already, leave
        # if (loss_L < done_threshold) and (loss_R < done_threshold):
        #     self.done = True
        #     self.reward = 0.0
        
        if self.action_counter >= self.max_episodes:
            # If all subgoals are done then episode is complete
            self.done = True
        else:
            # Otherwise move on to next subgoal
            self.goal_pressure_L = self.goal_pressure_sequence_L[self.action_counter]
            self.goal_pressure_R = self.goal_pressure_sequence_R[self.action_counter]

            self.done = False
            if self.use_step_loss == True:
                self.reward = -1.0
            
            if self.use_combined_loss == True:
                self.reward -= combined_loss          

        self.tim("Misc calculations")
        
        # prev_observation = self.step_observation # no longer important
        self.step_observation = self.calculate_step_observations() 
        observation = self.step_observation   
        # Stack 2 frames (can make convergence faster)
        # observation = np.concatenate((prev_observation, self.step_observation))
        
        return observation, self.reward, self.done, info

    def calculate_step_observations(self):
        # Calculate observation
        step_observation = np.array([
            self.pump.Lchamber.load_P, self.pump.Rchamber.load_P, self.pump.Lchamber.P, self.pump.Rchamber.P, 
            # self.pump.Lchamber.V, self.pump.Rchamber.V,  # V is not available in the real pump
            CHAMBER_LEN+self.pump.P_M, CHAMBER_LEN-self.pump.P_M, self.pump.P_M,
            self.pump.valve[0], self.pump.valve[1], self.pump.valve[2], self.pump.valve[3], self.pump.valve[4],  # converge faster but not stable
            self.pump.V_L, self.pump.V_R  # load
            # self.load_L, self.load_R  # load
            ])
        
        # Inject noise into observations (not affecting simulation state)
        obs_std = self.obs_noise*np.array([
            P_0, P_0, P_0, P_0,  
            CHAMBER_LEN, CHAMBER_LEN, 0,
            0, 0, 0, 0, 0,
            self.pump.V_L, self.pump.V_R
        ])
        noise = self.generate_observation_noise(noise_vector=obs_std)
        self.tim("Generate noise")
        step_observation += noise #np.sum([step_observation, noise], axis=0)
        self.tim("add noise")

        step_observation=np.concatenate((
            self.goal_pressure_sequence_L[self.action_counter:self.action_counter+self.future_frames], 
            self.goal_pressure_sequence_R[self.action_counter:self.action_counter+self.future_frames],
            step_observation))

        self.tim("np.concatenate")

        # print(noise/step_observation)
        # Normalize observation
        step_observation = self.normalization(step_observation)
        self.tim("normalize")

        return step_observation 

    def generate_moving_goal_pressure(self, amp_range, period=2*np.pi, phase=0):
    # def generate_moving_goal_pressure(self, range, period=2*np.pi, K_amp_noise=0, K_phase_noise=0):
        # assert K_amp_noise >= 0 and K_amp_noise <= 1
        # assert K_phase_noise >= 0 and K_phase_noise <= 1
        self.tim("Single start")
        amp_range=np.array(amp_range)
        x = np.arange(0, self.max_episodes+self.future_frames, 1)
        # phase_noise = np.random.uniform(0,2*np.pi, (1,)) 
        # amp_noise = np.random.uniform(-1,1, (self.max_episodes,))
        
        self.tim("Aranged")
        # y = np.sin(x/period + phase)

        wave_style=np.random.randint(3)
        if wave_style==0:
            y = np.sin(x/period + phase)
        if wave_style==1:
            y = signal.square(x/period) 
        if wave_style==2:
            y = signal.sawtooth(x/period) 
        # y = 0.5*((1-K_amp_noise)*np.sin(x/period + K_phase_noise*phase_noise) + K_amp_noise*amp_noise)
        # result will always be between -0.5 to -0.5, range of 1.0
        self.tim("Signaled")

        self.float_offset = 0.000005
        # assert amp_range[1] > amp_range[0]
        amp_range_ = (amp_range[1] - amp_range[0])/2
        mean = amp_range[0] + amp_range_
        y = mean + (amp_range_-self.float_offset)*y # mean + scaled y
        # assert np.min(y) >= amp_range[0] # min value should still be larger than range
        # assert np.max(y) <= amp_range[1] # max value should still be smaller than range
        self.tim("arithmteiced")

        return y

    def generate_composite_moving_goal_pressure(self, amp_range, n=3):
        def _generate_goal_pressure_sequence():
            return self.generate_moving_goal_pressure(
                amp_range= np.sort(np.random.uniform(amp_range[0],amp_range[1], 2)),
                phase = random.uniform(0,2*np.pi),
                period=random.uniform(3*np.pi, 7*np.pi)
                # period=random.uniform(1*np.pi, 5*np.pi)
                ) * P_0

        goal_pressure_sequence = _generate_goal_pressure_sequence()

        for i in range(1,n):
            goal_pressure_sequence = np.column_stack((goal_pressure_sequence, _generate_goal_pressure_sequence()))

        goal_pressure_sequence = np.average(goal_pressure_sequence, 1)
        self.tim("Average   ")

        return goal_pressure_sequence

    def generate_observation_noise(self, noise_vector=None):
        obs_size=noise_vector.shape
        if noise_vector.any() == None:
            noise=np.zeros(obs_size)
        # else:
        #     noise=np.random.normal(0, noise_vector)        # 0.038us unsure why
        noise=np.random.normal(0, 1, obs_size)*noise_vector #0.015us
        return noise

    def normalization(self, observation):
        self.observation_range_low = self.observation_range_low - self.float_offset
        self.observation_range_high = self.observation_range_high + self.float_offset

        norm_h, norm_l = 1.0, -1.0
        
        norm_observation = (observation - self.observation_range_low) / (self.observation_range_high - self.observation_range_low) * (norm_h - norm_l) + norm_l
        
        if ((norm_l <= norm_observation) & (norm_observation <= norm_h)).all():
            pass
        else:
            # print("obs:", observation)
            # print("norm_obs:", norm_observation)
            for i, (item_status, item_name) in enumerate(zip((norm_l <= norm_observation) & (norm_observation <= norm_h), self.observation_name)):
                if (item_status == False) and ("CHAMBER_LEN" not in item_name):
                   print(item_name, "observed: ", observation[i], " acceptable range: ", self.observation_range_low[i], self.observation_range_high[i])
                   if (norm_l <= norm_observation[i]):
                       print("clipping to min, {} -> {}".format(norm_observation[i], norm_l))
                       norm_observation[i] = norm_l
                   if (norm_observation[i] <= norm_h):
                       print("clipping to min, {} -> {}".format(norm_observation[i], norm_h))
                       norm_observation[i] = norm_h

            # raise Exception("Normalization error")
        return norm_observation


    def reset(self):
        ## Initialization
        self.done = False
        self.reward = 0
        self.prev_shaping = None
        self.action_counter = 0

        # Pump initialization with random load
        self.tim("Starting reset")

        self.load_L = random.uniform(self.load_range[0],self.load_range[1])
        self.load_R = random.uniform(self.load_range[0],self.load_range[1])
        self.pump = hybrid_pump(
            L_L=CHAMBER_LEN, L_R=CHAMBER_LEN, 
            load_chamber_ratio_L=self.load_L, 
            load_chamber_ratio_R=self.load_R,
            K_deform=self.K_deform
            )  # train with random load
        self.tim("create env")


        # Generate random sequence
        self.goal_pressure_sequence_L = self.generate_composite_moving_goal_pressure(self.goal_pressure_L_range)
        self.goal_pressure_sequence_R = self.generate_composite_moving_goal_pressure(self.goal_pressure_R_range)
        
        self.tim("Generate pressure sequence")
        # Get the first goal to get things started
        self.goal_pressure_L = self.goal_pressure_sequence_L[self.action_counter]
        self.goal_pressure_R = self.goal_pressure_sequence_R[self.action_counter]

        # Calculate observation
        self.step_observation = self.calculate_step_observations()
        observation = self.step_observation
        # observation = np.concatenate((self.step_observation, self.step_observation)) # stack two observations
        # try:
        #     observation = np.concatenate((self.prev_observation, self.step_observation)) # stack two observations
        
        # except:
        #     observation = np.concatenate((self.step_observation, self.step_observation)) # stack two observations

        # Calibrate
        self.tim("Step observation finished")
        V_L, V_R = self.calibrate()
        self.tim("Calibrate")

        # Reset pump to original state
        self.pump = hybrid_pump(
            L_L=CHAMBER_LEN, L_R=CHAMBER_LEN, 
            load_chamber_ratio_L=self.load_L, 
            load_chamber_ratio_R=self.load_R,
            K_deform = self.K_deform
            )  # train with random load

        ## Initialization
        self.done = False
        self.reward = 0
        self.prev_shaping = None
        self.action_counter = 0

        self.pump.V_L = V_L
        self.pump.V_R = V_R
        self.tim("Reset pump")

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
        observation=self.step_observation
        # observation = np.concatenate((self.step_observation, self.step_observation))
        return observation

    def calibrate(self, render=0):
        action_names = np.array([
            # Calibrate left load first. PC1 is P_0 as initialized
            "(a) Max right, open left valve to intake air ",
            "(b) Max left. (set PC1 to current reading)",
            "(c) Max left, open left load valve",
            # "(d) Take reading since now they're equalized to P2", 

            # Calibrate right load now. PC1 is P_0 as initialized
            "(a) Max left, open right valve to intake air ",
            "(b) Max right. (set PC1 to current reading)",
            "(c) Max right, open right load valve",
            # "(d) Take reading since now they're equalized to P2",

            ])

        # lL cL cR lR I  notation
        action_sequence = np.array([
        # Calibrate left load first. PC1 is P_0 as initialized
        [ 1, 0, 0, 1, 0, 0, 0,], # (a) Max right (optional), open left valve to intake air 
        [-1, 1, 0, 0, 0, 0, 0,], # (b) Max left. (set PC1 to current reading)
        [-1, 0, 1, 0, 0, 0, 0,], # (c) Max left, open left load valve (equalize load and chamber pressures)
        # Now the left load and left chamber pressure have equalized to P2 
        
        [-1, 0, 0, 0, 1, 0, 0,], # (a) Max left (optional), open right valve to intake air 
        [ 1, 1, 0, 0, 0, 0, 0,], # (b) Max right. (set PC1 to current reading)
        [ 1, 0, 0, 0, 0, 1, 0,], # (c) Max right, open right load valve (equalize load and chamber pressures)
        # Now the right load and right chamber pressure have equalized to P2 
        
        ])


	# 0,1 self.goal_pressure_L, float(self.goal_pressure_L < P_0),
        # 2,3 self.goal_pressure_R, float(self.goal_pressure_R < P_0),
        # 4,5 self.pump.Lchamber.load_P, self.pump.Rchamber.load_P, 
        # 6,7 self.pump.Lchamber.P, self.pump.Rchamber.P, 
        # # self.pump.Lchamber.V, self.pump.Rchamber.V,  # V is not available in the real pump
        # 9,10 CHAMBER_LEN + self.pump.P_M, CHAMBER_LEN - self.pump.P_M,
        # 11 self.pump.P_M,
        # 12, 13, 14, 15, 16 self.pump.valve[0], self.pump.valve[1], self.pump.valve[2], self.pump.valve[3], self.pump.valve[4],  # converge faster but not stable
        # 17, 18 self.load_L, self.load_R  # load

        # for i in range(self.max_episodes):
        for i, (name, action) in enumerate(zip(action_names, action_sequence)):
            _,_,_,_ = self.step(action)
            if render==1:
                self.render(0, title=name)
            if i == 1:
                P_L1 = self.pump.Lchamber.load_P
                P_C1 = self.pump.Lchamber.P
            if i == 2:
                assert self.pump.Lchamber.load_P == self.pump.Lchamber.P
                P_2 = self.pump.Lchamber.P # which should also be observation[5]
                V_L = (P_2-P_C1)/(P_L1-P_2) * self.pump.Lchamber.V
            if i == 4: # reading taking frames
                P_L1 = self.pump.Rchamber.load_P
                P_C1 = self.pump.Rchamber.P
            if i == 5:
                assert self.pump.Rchamber.load_P==self.pump.Rchamber.P
                P_2 = self.pump.Rchamber.P
                V_R = (P_2-P_C1)/(P_L1-P_2) * self.pump.Rchamber.V

        if render == 1:
            print("Calculated load volumes vs real: {:.07f}, {:.07f} | {:.07f}, {:.07f} ".format(
                V_L, V_R, self.pump.Lchamber.load_V, self.pump.Rchamber.load_V
        ))

        return V_L, V_R
        
    def print_step_info(self):
        print("Lchamber.P: {p1:.3f} Rchamber.P: {p2:.3f}| ".format(p1=self.pump.Lchamber.P/P_0, p2=self.pump.Rchamber.P/P_0),
              "Goal L {p1:.3f} | Goal R {p2:.3f} ".format(p1=self.goal_pressure_L/P_0, p2=self.goal_pressure_R/P_0),
            #   "Goal {p1} pressure {p2:.3f} | ".format(p1=self.goal_sequence_number, p2=self.goal_pressure/P_0),
              "Step reward: {p} | ".format(p=self.reward),
              "Error L: {p1:.3f} | Error R: {p2:.3f}".format(
                p1 = self.loss_L, # (self.pump.Lchamber.P - self.goal_pressure_L)/self.goal_pressure_L,
                p2 = self.loss_R # (self.pump.Rchamber.P - self.goal_pressure_R)/self.goal_pressure_R
                )
        )

    def render(self, time=0, title='', filename=None):
        # Render the environment to the screen
        return self.pump.render(time, title, filename)

    def tim(self, task_name):
        current_time = time.time()
        past_time = current_time - self.last_time
        # print("Time elapsed  {:.03f} for {}".format(past_time*1000, task_name))
        self.last_time = current_time

    def experiment_pump_properties(self, step_size=0.1):
        P_M_ =[]
        LCHAMBER_P_=[]
        RCHAMBER_P_=[]

        # Example action
        # (c) Max right, open right load valve (equalize load and chamber pressures)
        # [ 1, 0, 0, 0, 0, 1, 0,], 
        #   PM N  lL cL cR lR I  notation
        pm = 0
        def _left_to_right(title=''):        
            experiment_range = np.arange(-1, 1, step_size)
            # First loop: -0.08 to 0.08
            for pm in experiment_range:
                action = [pm, 1, 0, 0, 0, 0, 0,] # Open no valves, move to p_m 
                _,_,_,_ = self.step(action)
                self.render(0, title='{}, P_M={:.05f}'.format(title, self.pump.P_M))
                _print()
        
        def _right_to_left(title=''):
            experiment_range = -np.arange(-1, 1, step_size)
            # Second loop: 0.08 to -0.08
            for pm in experiment_range:
                action = [pm, 1, 0, 0, 0, 0, 0,] # Open no valves, move to p_m 
                _,_,_,_ = self.step(action)
                self.render(0, title='{}, P_M={:.05f}'.format(title, self.pump.P_M))
                _print()
        
        def _action(action, title=''):
            _,_,_,_ = self.step(action)
            self.render(0, title='{}, P_M={:.05f}'.format(title, self.pump.P_M))

        def _print():
            print("P_M, P_L, P_R | {: .05f} | {: 7.05f} | {: 7.05f}".format(
                self.pump.P_M, self.pump.Lchamber.P, self.pump.Rchamber.P
            ))
            P_M_.append(self.pump.P_M)
            LCHAMBER_P_.append(self.pump.Lchamber.P)
            RCHAMBER_P_.append(self.pump.Rchamber.P)
        
        # Pumping air into the right
        #         PM N  lL cL cR lR I  notation
        _action([-1, 0, 0, 0, 1, 0, 0,], title='Move left, open right valve') # Move left, open right valve

        # Eq all
        _action([-1, 0, 0, 1, 0, 0, 0,], title='Eq left') # Eq left
        _action([-1, 0, 0, 0, 1, 0, 0,], title='Eq right') # Eq right

        _left_to_right(title='1st loop pumping air into the right')
        _action([ 1, 0, 0, 1, 0, 0, 0,], title='Move right, open left valve')
        _action([-1, 0, 0, 0, 0, 0, 1,], title='Move left, open inner valve')
        _left_to_right(title='2nd loop pumping air into the right')


        # Pumping air into the left
        #         PM N  lL cL cR lR I  notation
        _action([ 1, 0, 0, 1, 0, 0, 0,], title='Move right, open left valve') 

        # Eq all
        _action([ 1, 0, 0, 1, 0, 0, 0,], title='Eq left') # Eq left
        _action([ 1, 0, 0, 0, 1, 0, 0,], title='Eq right') # Eq right

        _right_to_left(title='1st loop pumping air into the left')
        _action([-1, 0, 0, 0, 1, 0, 0,], title='Move left, open right valve')
        _action([ 1, 0, 0, 0, 0, 0, 1,], title='Move right, open inner valve')
        _right_to_left(title='2nd loop pumping air into the left')


        # Pumping air out of the right
        #         PM N  lL cL cR lR I  notation
        _action([ 1, 0, 0, 0, 1, 0, 0,], title='Move right, open right valve') 

        # Eq all
        _action([ 1, 0, 0, 1, 0, 0, 0,], title='Eq left') # Eq left
        _action([ 1, 0, 0, 0, 1, 0, 0,], title='Eq right') # Eq right

        _right_to_left(title='1st loop pumping air out of the right')
        _action([-1, 0, 0, 1, 0, 0, 0,], title='Move left, open right valve')
        _action([ 1, 0, 0, 0, 0, 0, 1,], title='Move right, open inner valve')
        _right_to_left(title='2nd loop pumping air out of the right')

        # Pumping air out of the left
        #         PM N  lL cL cR lR I  notation
        _action([-1, 0, 0, 1, 0, 0, 0,], title='Move left, open left valve') 

        # Eq all
        _action([-1, 0, 0, 1, 0, 0, 0,], title='Eq left') # Eq left
        _action([-1, 0, 0, 0, 1, 0, 0,], title='Eq right') # Eq right

        _left_to_right(title='1st loop pumping air out of the left')
        _action([ 1, 0, 0, 0, 1, 0, 0,], title='Move left, open right valve')
        _action([-1, 0, 0, 0, 0, 0, 1,], title='Move right, open inner valve')
        _left_to_right(title='2nd loop pumping air out of the left')

        pickle.dump([P_M_, LCHAMBER_P_, RCHAMBER_P_], open( "./scripts/chamber_experiments.p", "wb" ) )

if "__main__" == __name__:

    env = PumpEnvVar_Two(
        load_range=[0, 1.0], 
        goal_pressure_R_range=[0.6, 1.6],
        goal_pressure_L_range=[0.6, 1.6],
        max_episodes=10000,
        use_combined_loss = True,
        use_step_loss = False,
        )

        # env.reset()
        # for i in range(env.max_episodes):
        #     print(i)
    env.experiment_pump_properties()

    [P_M_, LCHAMBER_P_, RCHAMBER_P_] = pickle.load( open( "./scripts/chamber_experiments.p", "rb" ) )
    P_M_=np.array(P_M_) 
    LCHAMBER_P_=np.array(LCHAMBER_P_) 
    RCHAMBER_P_=np.array(RCHAMBER_P_)

    fig, ax = plt.subplots(ncols=1, nrows=3, figsize=(20, 5))
    ax=ax.flatten()
    ax[0].plot(P_M_)
    ax[1].plot(LCHAMBER_P_)
    ax[2].plot(RCHAMBER_P_)
    plt.show()
