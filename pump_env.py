import gym
from gym import spaces
import numpy as np
import random
from pump_sim_dis import hybrid_pump
# from pump_sim_load import hybrid_pump


P_0 = 1.01*1e5  # Pa
MAX_STEP = 100
CHAMBER_LEN = 0.1

# This is the basic env

class PumpEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render.modes": ["human"]}

    def __init__(self, goal_pressure_range=[1.1, 4.0]):
        super(PumpEnv, self).__init__()
        # Continuous actions
        self.action_space = spaces.Box(low=-1, high=1,
                                            shape=(2,), dtype=np.float32)
        # Input observation:
        dim_obs = 11
        n_stack_obs = 2
        dim_stack_obs = int(dim_obs * n_stack_obs)
        self.observation_space = spaces.Box(low=-1.0, high=1.0,
                                            shape=(dim_stack_obs,), dtype=np.float32)
        self.pump = hybrid_pump(CHAMBER_LEN, CHAMBER_LEN)
        self.action_counter = 0
        self.reward = 0
        self.goal_pressure_range = goal_pressure_range
        self.goal_pressure = 0
        self.prev_observation = np.zeros((dim_obs,))
        self.load = 0.0


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
                                            self.load  # load
                                            ])
        # Normalize observation
        self.step_observation = self.normalization(self.step_observation)
        # Stack 2 frames (can make convergence faster)
        observation = np.concatenate((prev_observation, self.step_observation))
        
        return observation, self.reward, self.done, info


    def normalization(self, observation):
        # Normalize observation
        goal_pressure_min = 0.05 * P_0
        goal_pressure_max = 10 * P_0
        val_L_min = 0.0
        val_L_max = 2.0
        float_offset = 0.001
        # observation_range_low = np.array([goal_pressure_min, 0.0, 0.05*P_0, 0.05*P_0, 6.031857e-05, 6.031857e-05, -0.05])
        # observation_range_high = np.array([goal_pressure_max, 1.0, 10*P_0, 10*P_0, 0.000185983, 0.000313, +0.05])
        # observation_range_low = np.array([goal_pressure_min, 0.0, 0.05*P_0, 0.05*P_0, 6.031857e-05, 6.031857e-05, -0.05])
        # observation_range_high = np.array([goal_pressure_max, 1.0, 10*P_0, 10*P_0, 0.0002626, 0.000440, +0.05])
        # observation_range_low = np.array([goal_pressure_min, 0.0, 0.05*P_0, 0.05*P_0, 0.049, 0.049, -0.05])
        # observation_range_high = np.array([goal_pressure_max, 1.0, 10*P_0, 10*P_0, 0.151, 0.151, +0.05])
        observation_range_low = np.array([goal_pressure_min, 0.0, 0.01*P_0, 0.01*P_0, 0.049, 0.049, -0.05, 0.0, 0.0, 0.0, val_L_min])
        observation_range_high = np.array([goal_pressure_max, 1.0, 10*P_0, 10*P_0, 0.151, 0.151, +0.05, 1.0, 1.0, 1.0, val_L_max])
        # observation_range_low = np.array([goal_pressure_min, 0.0, 0.01*P_0, 0.01*P_0, 0.049, 0.049, -0.05, 0.0, 0.0, 0.0])
        # observation_range_high = np.array([goal_pressure_max, 1.0, 10*P_0, 10*P_0, 0.151, 0.151, +0.05, 1.0, 1.0, 1.0])
        
        observation_range_low = observation_range_low - float_offset
        observation_range_high = observation_range_high + float_offset
        
        norm_h, norm_l = 1.0, -1.0

        
        norm_observation = (observation - observation_range_low) / (observation_range_high - observation_range_low) * (norm_h - norm_l) + norm_l
        if ((norm_l <= norm_observation) & (norm_observation <= norm_h)).all():
            pass
        else:
            print("obs:", observation)
            print("norm_obs:", norm_observation)
            raise Exception("Normalization error")
        return norm_observation

    
    def reset(self):
        ## Initialization
        self.done = False
        self.reward = 0
        self.prev_shaping = None
        self.action_counter = 0

        # Pump initialization
        self.pump = hybrid_pump(CHAMBER_LEN, CHAMBER_LEN)
        
        # Set goal pressure (0.7~2.0*P_0)
        self.goal_pressure = random.uniform(self.goal_pressure_range[0], self.goal_pressure_range[1]) * P_0

        # Calculate observation
        self.step_observation = np.array([self.goal_pressure, float(self.goal_pressure < P_0),
                                self.pump.Lchamber.P, self.pump.Rchamber.P, 
                                # self.pump.Lchamber.V, self.pump.Rchamber.V, 
                                CHAMBER_LEN + self.pump.P_M, CHAMBER_LEN - self.pump.P_M,  # chamber length
                                self.pump.P_M,
                                self.pump.valve[0], self.pump.valve[1], self.pump.valve[2],
                                self.load  # load
                                ])
        # Normalize observation
        self.step_observation = self.normalization(self.step_observation)
        # Stack 2 frames
        observation = np.concatenate((self.step_observation, self.step_observation))

        return observation  # reward, done, info can't be included


    def set_goal_pressure(self, goal_pressure):
        self.goal_pressure = goal_pressure


    def render(self, time=0):
        # Render the environment to the screen
        self.pump.render(time)


    def print_info(self):
        print("Total Gas: {:.06f} | Gasses {:.06f}, {:.06f} | Total Volume {:.06f} | Pressures {:.06f}, {:.06f} | Volumes  {:.06f}, {:.06f}".format(
        self.pump.Lchamber.V*self.pump.Lchamber.P + self.pump.Rchamber.V*self.pump.Rchamber.P,
        self.pump.Lchamber.V*self.pump.Lchamber.P, self.pump.Rchamber.V*self.pump.Rchamber.P,
        self.pump.Lchamber.V+self.pump.Rchamber.V, 
        self.pump.Lchamber.P, self.pump.Rchamber.P,
        self.pump.Lchamber.V, self.pump.Rchamber.V))

if "__main__" == __name__:
    CM_2_M = 0.01
    env = PumpEnv()
    env.reset()
    env.print_info()
    # env.render()

    for i in range(10):
        action=[random.uniform(-1,1), 0.3]
        env.step(action)
        env.print_info()
        env.render()
        
        # action=[1.0, 0.3]
        # env.step(action)
        # env.print_info()
        # # env.render()

    action=[0, 0.3]
    env.step(action)
    env.print_info()
    
