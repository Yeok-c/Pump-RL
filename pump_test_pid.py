from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, DDPG, TD3, SAC
from pump_realenv_variable_load_two_DRA import PumpRealEnvVar_Two
from pump_env_variable_load_two_DRA import PumpEnvVar_Two as PumpSimEnvVar_Two
from curi_communication_udp import curi_communication_udp
import pickle
import os
import cv2
import numpy as np
from simple_pid import PID

# os.chdir("Pump-RL-Real")

P_0 = 1.01*1e5  # Pa
kPa = 1e3



# self, kp = -0.06, ki=-0.012, kd=-0.01):
# self, kp = 0.06, ki=0.012, kd=0.01):
class Pump_PID():
    def __init__(
        self, kp = 0.04, ki=0.012, kd=0.01):
        self.pid = PID(kp, ki, kd, setpoint=0)
        self.pid.output_limits = (-2, 2)    # Output value will be between 0 and 10
        self.p_lim = 1

    def calc_Mmax(self, p_m, render=False):
        M_min = -self.p_lim-p_m
        M_max =  self.p_lim-p_m
        if render==True:
            print(f"p_m: {p_m:.05f}, M_max: {M_max:.05f}, M_min: {M_min:.05f}")
        return M_max, M_min

    def controller(self, error, p_m, goal, side="left"):

        def _mirror_actions(action):
            action_ = np.array(action)
            action_[0] =-action[0]
            action_[1] = action[1] # "None" valve
            action_[2] = action[5] 
            action_[3] = action[4]
            action_[4] = action[3]
            action_[5] = action[2]
            action_[6] = action[6] # Inner valve
            return action_

        p_m = p_m/0.008 # relative position within [-1, 1]
        M_max, M_min = self.calc_Mmax(p_m)
        control_signal= self.pid(error)
        print(f"Control signal | error : {control_signal:.05f} | {error:.05f}")
        action_queue=[]


        # if control_signal >= 2: # error > 40:
        # # Very large errors
        #     if goal <= -30: # error > 40:
        #     # Very large errors
        #         print(f"cocking twice for negative pressure {control_signal:.05f}")
        #         action_queue=[
        #             [ 1, 0, 0, 0, 1, 0, 0],
        #             [-1, 0, 0, 0, 0, 0, 1],
        #             [ 1, 0, 0, 0, 1, 0, 0],
        #             [-1, 0, 0, 0, 0, 0, 1],
        #             [ 1, 0, 0, 0, 1, 0, 0],
        #             [-1, 0, 0, 0, 0, 0, 1],
        #             [ 0, 0, 1, 0, 0, 0, 0],
        #         ]
        #     else:
        #         print(f"cocking twice for negative pressure {control_signal:.05f}")
        #         action_queue = [
        #             [ 1, 0, 0, 0, 1, 0, 0],
        #             [-1, 0, 0, 0, 0, 0, 1],
        #             [ 0, 0, 1, 0, 0, 0, 0],
        #         ]

            
        # # if error < -20:
        # #     print(f"cocking for negative pressure {control_signal:.05f}")
        # #     action_queue = [
        # #         [ 1, 0, 0, 0, 1, 0, 0],
        # #         [-1, 0, 0, 0, 0, 0, 1],
        # #         [ 0, 0, 1, 0, 0, 0, 0],
        # #     ]


        # elif control_signal <= -2: #error < -40:
        #     if goal >= 30:
                
        #         print(f"cocking twice for positive pressure {control_signal:.05f}")
        #         action_queue = [
        #             [-1, 0, 0, 0, 1, 0, 0],
        #             [ 1, 0, 0, 0, 0, 0, 1],
        #             [-1, 0, 0, 0, 1, 0, 0],
        #             [ 1, 0, 0, 0, 0, 0, 1],
        #             [-1, 0, 0, 0, 1, 0, 0],
        #             [ 1, 0, 0, 0, 0, 0, 1],
        #             [ 0, 0, 1, 0, 0, 0, 0],
        #         ]
        #     else: 

        #         print(f"cocking twice for positive pressure {control_signal:.05f}")
        #         action_queue = [
        #             [-1, 0, 0, 0, 1, 0, 0],
        #             [ 1, 0, 0, 0, 0, 0, 1],
        #             [ 0, 0, 1, 0, 0, 0, 0],
        #         ]

        # # elif error > 20: # else:# control_signal > M_max and control_signal < 1:
        # #     print(f"cocking once for positive pressure {control_signal:.05f}")
        # #     action_queue = [
        # #         [-1, 0, 0, 0, 1, 0, 0],
        # #         [ 1, 0, 0, 0, 0, 0, 1],
        # #         [ 0, 0, 1, 0, 0, 0, 0],
        # #     ]

        # # Small errors that only require minor adjustments
        # elif control_signal < M_max and control_signal > 0: # control signal is within limits
        #     print(f"negative pressure simple {control_signal:.05f}")
        #     if side == "left":
        #         p_m += control_signal
        #     else:
        #         p_m -= control_signal    
        # elif control_signal > M_min and control_signal < 0: # control signal is within limits
        #     print(f"positive pressure is simple {control_signal:.05f}")
        #     if side == "left":
        #         p_m += control_signal
        #     else:
        #         p_m -= control_signal
        
        # # Medium errors that require cocking
        # elif control_signal >= M_max and control_signal > 0: # control signal is within limits
        #     print(f"reset pm for negative action {control_signal:.05f}")
        #     action_queue = [
        #         [-1, 0, 0, 1, 0, 0, 0],
        #         [ 0, 0, 1, 0, 0, 0, 0],
        #     ]
        
        # elif control_signal <= M_min and control_signal < 0: # control signal is within limits
        #     print(f"reset pm for positive action {control_signal:.05f}")
        #     action_queue = [
        #         [ 1, 0, 0, 1, 0, 0, 0],
        #         [ 0, 0, 1, 0, 0, 0, 0],
        #     ]

        p_m += control_signal
        if action_queue !=[] and side=="right":
            action_queue = _mirror_actions(action_queue)


        return p_m, action_queue





if "__main__" == __name__:
    udp = curi_communication_udp("127.0.0.1", 13331, "127.0.0.1", 13332)
    udp.open()
    print("Open udp")

    MEAN_REWARD=[]
    STD_REWARD=[]
    # for load in test_loads:

    PID = Pump_PID()
    episodes = 10
    for ep in range(3,7):
        # Create environment
        load = 0
        env = PumpRealEnvVar_Two(
            # load_range=[0.1, 0.1], 
            load_range=[load, load], 
            goal_pressure_R_range=[0.5, 1.7],
            goal_pressure_L_range=[0.5, 1.7],
            max_episodes=100,
            use_combined_loss = True,
            use_step_loss = False,   
            # obs_noise = 0.01,
            # K_deform = 0.01,
            udp=udp
            )

        env_sim = PumpSimEnvVar_Two(
            # load_range=[0.1, 0.1], 
            load_range=[load, load], 
            goal_pressure_R_range=[0.5, 1.7],
            goal_pressure_L_range=[0.5, 1.7],
            max_episodes=100,
            use_combined_loss = True,
            use_step_loss = False,   
            # obs_noise = 0.01,
            # K_deform = 0.01,
            )
        load = str(load).replace('.', '_')

        # env.reset()

        GL=env.goal_pressure_sequence_L
        GR=env.goal_pressure_sequence_R


        P_L, P_R, P_L_S, P_R_S, P_R_G, P_L_G, R, R_S = [], [],[],[],[], [], [], []
        
        # Start simulation
        if not os.path.exists(f"./saved_figures/{load}"):
            os.system(f"mkdir ./saved_figures/{load}")
        if not os.path.exists(f"./saved_figures/{load}/{ep}"):
            os.system(f"mkdir ./saved_figures/{load}/{ep}")

        P_L, P_R, P_L_S, P_R_S, P_R_G, P_L_G, R, R_S = [], [],[],[],[], [], [], []
        [_, _, _, _, P_R_G, P_L_G, _, _, _, _] = pickle.load( open( f"./saved_figures/0/tracking_results_ep_{ep}.p", "rb" ) )
        env.goal_pressure_sequence_L = P_L_G*1000+P_0 
        env.goal_pressure_sequence_R = np.ones((env.max_episodes+env.future_frames,))*P_0 # P_R_G*1000+P_0


        # # Start simulation
        # if not os.path.exists(f"./saved_figures/{load}"):
        #     os.system(f"mkdir ./saved_figures/{load}")
        # if not os.path.exists(f"./saved_figures/{load}/{ep}"):
        #     os.system(f"mkdir ./saved_figures/{load}/{ep}")

        env.render(
            title="step: {}".format(env.action_counter), 
            filename="./saved_figures/{}/{}/step_{:03d}.png".format(load, ep, env.action_counter),
            time=1
        )

        P_R_G = (env.goal_pressure_sequence_R-P_0)/1000
        P_L_G = (env.goal_pressure_sequence_L-P_0)/1000
        done = False


        # match sim properties to real env
        # env.pump.V_L = env.pump.V_L *2
        # env.pump.V_R = env.pump.V_R *2
        env_sim.pump.V_L = env.pump.V_L
        env_sim.pump.V_R = env.pump.V_R
        env_sim.goal_pressure_sequence_L = env.goal_pressure_sequence_L
        env_sim.goal_pressure_sequence_R = env.goal_pressure_sequence_R
        VL, VR = env.pump.V_L/env.pump.Lchamber.V, env.pump.V_R/env.pump.Rchamber.V

        obs, reward, done, info = env.step([0, 1, 0,0,0,0,0])
        P_M=0
        action_queue = []
        while not done:
            env_filename ="./saved_figures/{}/{}/step_{:03d}.png".format(load, ep, env.action_counter) 
            tracking_filename = "./saved_figures/{}/{}/tracking_step_{:03d}.png".format(load, ep, env.action_counter)
            env_render = env.render(
                title="step: {}".format(env.action_counter), 
                # filename=env_filename,
                time=1
            )

            # self.Lchamber.load_P =   self.pressure[2]
            # self.Rchamber.load_P =   self.pressure[3]
            
            P_L.append((env.pump.pressure[2]-P_0)/1000)
            P_R.append((env.pump.pressure[3]-P_0)/1000)
            P_L_S.append((env_sim.pump.Lchamber.load_P-P_0)/1000)
            P_R_S.append((env_sim.pump.Rchamber.load_P-P_0)/1000)
            R.append(env.reward)
            R_S.append(env_sim.reward)

            # env.goal_pressure_sequence_L/1000
            # env.goal_pressure_sequence_R/1000
            pickle.dump([P_L, P_R, P_L_S, P_R_S, P_R_G, P_L_G, R, R_S, VL, VR], open( "./scripts/save_sim_and_real.p", "wb" ) )
            # pickle.dump([P_L, P_R, P_R_G, P_L_G, R, VL, VR], open( "./scripts/save.p", "wb" ) )
            os.system("python ./scripts/plot_goals_two_sim_and_real.py")
            # cv2.imshow('Goals vs achieved: Trained on sequential goals', cv2.imread('./scripts/file.png'))
            os.system("cp ./scripts/file.png {}".format(tracking_filename))
            
            A = env_render # cv2.imread(env_filename)
            A = A[50:420,:,:]
            B = cv2.imread(tracking_filename)
            C = np.concatenate((A,B), axis=0)
            cv2.imwrite(env_filename, C)



            # action, _states = model.predict(obs)

            error_L = (env.goal_pressure_L - env.pump.pressure[2])/kPa
            error_R = (env.goal_pressure_R - env.pump.pressure[3])/kPa

            if action_queue == []: # if no queue
                # if no queue, get current action
                pm, action_queue = PID.controller(error_L, env.pump.P_M, env.goal_pressure_L/kPa)
                print(f"Just got actions, pm: {pm:.05f}, queue:{action_queue}")

            if action_queue == []: # if no queue
                # Essentially do not move, just open valve
                action = [ pm, 0, 1, 0, 0, 0, 0,]

            else: # If received action queue
                print("Current action queue: ", action_queue)
                action = action_queue[0]
                action_queue = action_queue[1:] # remove zero-th element


            # print("\n\n\n")
            # for idx in range(len(env.step_observation)):
            #     print("{:02d}, sim, real: {:.6f}, {:.6f}, {}".format(idx, env_sim.step_observation[idx], env.step_observation[idx], env.observation_name[idx] ))
            # print("\n\n\n")

            obs, reward, done, info = env.step(action)
            _, _, _, _ = env_sim.step(action)
            # print("Rchamber.P: {p:.3f} | ".format(p=env.pump.Rchamber.P/P_0),
            #       "Goal {p1} pressure {p2:.3f} | ".format(p1=env.goal_sequence_number, p2=env.goal_pressure/P_0),
            #       "Step reward: {p} | ".format(p=reward),
            #       "Error: {p:.3f} | ".format(
            #         p = (env.pump.Rchamber.P - env.goal_pressure)/env.goal_pressure
            #         # p = abs((env.pump.Rchamber.P - env.goal_pressure)/env.goal_pressure)
            #         )
            # )

        
        pickle.dump([P_L, P_R, P_L_S, P_R_S, P_R_G, P_L_G, R, R_S, VL, VR], open( "./scripts/save_sim_and_real.p", "wb" ) )
        os.system("python ./scripts/plot_goals_two_sim_and_real.py")
        cv2.imshow('Goals vs achieved: Trained on sequential goals', cv2.imread('./scripts/file.png'))
        cv2.waitKey(1)

        os.system("cp ./scripts/file.png ./saved_figures/{}/tracking_results_ep_{}.png".format(load, ep))
        os.system("rm -rf ./saved_figures/*/*/tracking_step_*")
        os.system("cp ./scripts/save_sim_and_real.p ./saved_figures/{}/tracking_results_ep_{}.p".format(load, ep))
        os.system("ffmpeg -r 3 -i ./saved_figures/{p1}/{p2}/step_%03d.png -vcodec mpeg4 -y ./saved_figures/{p1}/tracking_results_ep_{p2}.mp4".format(p1=load, p2=ep))

        # cv2.waitKey(0)

    pickle.dump([test_loads, MEAN_REWARD,STD_REWARD], open( "./scripts/load_test_results.p", "wb" ) )
    os.system("python ./scripts/plot_load_test_results.py")
    # cv2.imshow('Mean reward with different loads, evaluated with 1000 episodes', cv2.imread('./scripts/file.png'))
    # cv2.waitKey(0)

    os.system("cp ./scripts/file.png ./saved_figures/overall_results.png")
