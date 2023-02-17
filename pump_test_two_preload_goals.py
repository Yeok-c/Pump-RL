from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, DDPG, TD3, SAC
from pump_realenv_variable_load_two_DRA import PumpRealEnvVar_Two
from pump_env_variable_load_two_DRA import PumpEnvVar_Two as PumpSimEnvVar_Two
from curi_communication_udp import curi_communication_udp
import pickle
import os
import cv2
import numpy as np
import glob

# os.chdir("Pump-RL-Real")

P_0 = 1.01*1e5  # Pa

udp = curi_communication_udp("127.0.0.1", 13331, "127.0.0.1", 13332)
udp.open()
print("Open udp")

MEAN_REWARD=[]
STD_REWARD=[]
# for load in test_loads:

# episodes=6
# for ep in range(episodes):
for ep in [1, 2, 3, 4, 5]:
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
    

    # Load the trained agent

    model_dir = "models"
    model_run = "1675862353"
    model_step = "16000000"    

    model_path = f"{model_dir}/{model_run}/{model_step}"  # for var load experiment
    model = SAC.load(model_path, env=env, print_system_info=True)


    P_L, P_R, P_L_S, P_R_S, P_R_G, P_L_G, R, R_S = [], [],[],[],[], [], [], []
    [_, _, _, _, P_R_G, P_L_G, _, _, _, _] = pickle.load( open( f"./saved_figures/0/tracking_results_ep_{ep}.p", "rb" ) )
    env.goal_pressure_sequence_L = P_L_G*1000+P_0 
    env.goal_pressure_sequence_R = np.ones((env.max_episodes+env.future_frames,))*P_0 # P_R_G*1000+P_0


    # Start simulation
    if not os.path.exists(f"./saved_figures/{load}"):
        os.system(f"mkdir ./saved_figures/{load}")
    if not os.path.exists(f"./saved_figures/{load}/{ep}"):
        os.system(f"mkdir ./saved_figures/{load}/{ep}")

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

        # print("obs:", obs[-1])
        # print('env.load', env.load)
        
        action, _states = model.predict(env.step_observation)
        # while np.argmax(action)==4 or np.argmax(action)==5:
        # action, _states = model.predict(env.step_observation)
        # print(action)
        # action, _states = model.predict(env_sim.step_observation)
        print("\n\n\n")
        for idx in range(len(env.step_observation)):
            print("{:02d}, sim, real: {:.6f}, {:.6f}, {}".format(idx, env_sim.step_observation[idx], env.step_observation[idx], env.observation_name[idx] ))
        print("\n\n\n")
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