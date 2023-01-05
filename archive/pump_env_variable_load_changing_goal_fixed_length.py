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

class PumpEnvVar_Sequence(PumpEnv):
    def __init__(self, load_range=[0.0,2.0], 
    goal_pressure_range=[1.1, 2.0], 
    goal_sequence_length = 10,
    episode_length = MAX_STEP
    ):
        # super(PumpEnv, self).__init__()
        # Continuous actions
        self.action_space = spaces.Box(low=-1, high=1,
                                            shape=(5,), dtype=np.float32)
        # Input observation:
        dim_obs = 10
        n_stack_obs = 2
        n_stack_calib = 0
        dim_stack = int(dim_obs * n_stack_obs + dim_obs * n_stack_calib)
        self.observation_space = spaces.Box(low=-1.0, high=1.0,
                                            shape=(dim_stack,), dtype=np.float32)
        self.load_range = load_range
        self.load = random.uniform(self.load_range[0],self.load_range[1])
        self.pump = hybrid_pump(L_L=CHAMBER_LEN, L_R=CHAMBER_LEN, load=self.load)
        self.action_counter = 0
        self.reward = 0

        self.goal_sequence_length = goal_sequence_length
        self.goal_pressure_range = goal_pressure_range
        self.goal_pressure_sequence = np.zeros((self.goal_sequence_length,))
        self.goal_sequence_number = 0
        self.goal_pressure = self.goal_pressure_sequence[self.goal_sequence_number]
        self.prev_observation = np.zeros((dim_obs,))
        self.calibration_observations = np.zeros((n_stack_calib*dim_obs,)),
        self.episode_length = episode_length

    def step(self, action):
        # Execute one time step within the environment
        self.action_counter += 1

        # Reset valve for visualization
        self.pump.close_R_valve()
        self.pump.close_L_valve()
        self.pump.close_inner_valve()

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
        
        # [Action 1]: Valve  (Change P_M first)

        if self.action_space.shape==(2,):
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
        
        if self.action_space.shape==(5,):
            if action[1] > 0:
                self.pump.open_R_valve()
            if action[2] > 0:
                self.pump.open_inner_valve()
            if action[3] > 0:
                self.pump.open_L_valve()

        # Check if the pump is done
        info = {}
        done_threshold = 0.01
        loss_R = abs(self.pump.Rchamber.P - self.goal_pressure)/self.goal_pressure
        # if loss_R < done_threshold:
        #     self.done = False
        #     # Current subgoal complete
        #     self.reward = 2.0 
        #     self.goal_sequence_number += 1 
        #     if self.goal_sequence_number >= self.goal_sequence_length:
        #         # If all subgoals are done then episode is complete
        #         self.done = True
        #     else:
        #         # Otherwise move on to next subgoal
        #         self.goal_pressure = self.goal_pressure_sequence[self.goal_sequence_number]

        self.goal_sequence_number += 1 
        if self.goal_sequence_number >= self.goal_sequence_length:
            # If all subgoals are done then episode is complete
            self.done = True
        else:
            # Otherwise move on to next subgoal
            self.goal_pressure = self.goal_pressure_sequence[self.goal_sequence_number]

        if self.action_counter > self.episode_length:
            self.done = True
            # self.reward = -1.0
            info["episode_timeout"] = True
        else:
            # do not punish for additional steps
            self.done = False
            # self.reward = -1.0

            # instead punish for loss
            self.reward = -loss_R
    
        # Calculate observation
        prev_observation = self.step_observation
        self.step_observation = self.calculate_step_observations()
    
        # Stack 2 frames (can make convergence faster)
        observation = np.concatenate((prev_observation, self.step_observation))
        
        return observation, self.reward, self.done, info

    def calculate_step_observations(self):
        step_observation = np.array([self.goal_pressure, float(self.goal_pressure < P_0),
                                            self.pump.Lchamber.P, self.pump.Rchamber.P, 
                                            # self.pump.Lchamber.V, self.pump.Rchamber.V,  # V is not available in the real pump
                                            CHAMBER_LEN + self.pump.P_M, CHAMBER_LEN - self.pump.P_M,
                                            self.pump.P_M,
                                            self.pump.valve[0], self.pump.valve[1], self.pump.valve[2],  # converge faster but not stable
                                            # self.load  # load
                                            ])
        # Normalize observation
        return self.normalization(step_observation)
    
    # def calculate_calibration_observations(self):
    #     self.step(np.array([0.5, 0, 0, 0]))
    #     step_observation_1 = self.calculate_step_observations()
    #     self.step(np.array([-0.5, 0, 0, 0]))
    #     step_observation_2 = self.calculate_step_observations()
    #     # Newer first
    #     self.calibration_observations = np.concatenate((step_observation_2, step_observation_1))

    def generate_moving_goal_pressure(self, range, period=2*np.pi, K_amp_noise=0, K_phase_noise=0):
        assert K_amp_noise >= 0 and K_amp_noise <= 1
        assert K_phase_noise >= 0 and K_phase_noise <= 1
        x = np.arange(0, self.goal_sequence_length, 1)
        phase_noise = np.random.uniform(0,2*np.pi, (1,)) 
        amp_noise = np.random.uniform(-1,1, (self.goal_sequence_length,))
        y = 0.5*((1-K_amp_noise)*np.sin(x/period + K_phase_noise*phase_noise) + K_amp_noise*amp_noise)
        # result will always be between -0.5 to -0.5, range of 1.0

        assert range[1] > range[0]
        range_ = (range[1] - range[0])/2
        mean = range[0] + range_
        y = mean + range_*y # mean + scaled y
        assert np.min(y) > range[0] # min value should still be larger than range
        assert np.max(y) < range[1] # max value should still be smaller than range
        return y

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
        # self.goal_pressure = random.uniform(self.goal_pressure_range[0], self.goal_pressure_range[1]) * P_0

        goal_pressure_sequence_1 = self.generate_moving_goal_pressure(
            self.goal_pressure_range, 
            K_phase_noise = 1,
            period=2*np.pi
            ) * P_0
        
        # curl smartgrasping@10.30.7.117:~/pump/models/1670924068/6000000.zip
        goal_pressure_sequence_2 = self.generate_moving_goal_pressure(
            self.goal_pressure_range, 
            K_phase_noise = 1,
            period=3*np.pi
            ) * P_0

        self.goal_pressure_sequence = (goal_pressure_sequence_1 + goal_pressure_sequence_2)/2

        self.goal_sequence_number = 0
        self.goal_pressure = self.goal_pressure_sequence[self.goal_sequence_number]

        # Calculate observation
        self.step_observation = self.calculate_step_observations()
        observation = np.concatenate((self.step_observation, self.step_observation)) # stack two observations

        # Calibrate
        # self.calculate_calibration_observations()
        # observation = np.concatenate((self.calibration_observations, self.calibration_observations)) # stack two observations
        
        return observation
    

    def set_load_and_goal_pressure(self, load, goal_pressure):
        # Specify load and goal for testing
        self.load = load
        self.pump = hybrid_pump(L_L=CHAMBER_LEN, L_R=CHAMBER_LEN, load=load)
        self.goal_pressure = goal_pressure
        self.step_observation = self.calculate_step_observations()
        observation = np.concatenate((self.step_observation, self.step_observation))
        return observation

    
    def print_step_info(self):
        print("Lchamber.P: {p1:.3f} Rchamber.P: {p2:.3f}| ".format(p1=self.pump.Lchamber.P/P_0, p2=self.pump.Rchamber.P/P_0),
            #   "Goal L {p1:.3f} | Goal R {p2:.3f} ".format(p1=self.goal_pressure_L/P_0, p2=self.goal_pressure_R/P_0),
              "Goal {p1} pressure {p2:.3f} | ".format(p1=self.goal_sequence_number, p2=self.goal_pressure/P_0),
              "Step reward: {p} | ".format(p=self.reward),
              "Error R: {p2:.3f}".format(
                p2 = (self.pump.Rchamber.P - self.goal_pressure)/self.goal_pressure
                )
        )

if "__main__" == __name__:
    env = PumpEnvVar_Sequence(goal_sequence_length = 10)
    obs = env.reset()
    print('obs', obs)
    # print(env.var_L)
    # env.render()
    
    for i in range(5):
        action = env.action_space.sample()
        obs, _, _, _ = env.step(action)
        env.render(0)

    # action=[+1.0, -1.0]
    # obs, reward, done, info = env.step(action)
    # print('obs', obs)
    # env.render()
    # obs = env.reset()
    # print('obs', obs)

    # for i in range(5):
    #     action = env.action_space.sample()
    #     obs, _, _, _ = env.step(action)
    #     print('obs:', obs[-1], env.var_L)
    #     env.render(0)

    # env.set_load_and_goal_pressure(0.01, 1.1*P_0)
    # for i in range(5):
    #     action = env.action_space.sample()
    #     obs, _, _, _ = env.step(action)
    #     print('obs:', obs[-1], env.var_L)
    #     env.render(0)