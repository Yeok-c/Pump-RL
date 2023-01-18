from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, DDPG, TD3, SAC
from pump_env_variable_load_two_DRA import PumpEnvVar_Two
import pickle
import os
import cv2
import numpy as np

P_0 = 1.01*1e5  # Pa

test_loads=[
    # 0, 0.5, 
    1, 2, 
    # 3, 4, 5, 6, 7, 8
]

os.system(f"rm -rf ./saved_figures")
os.system(f"mkdir ./saved_figures")

MEAN_REWARD=[]
STD_REWARD=[]
for load in test_loads:
    # Create environment
    env = PumpEnvVar_Two(
        # load_range=[0.1, 0.1], 
        load_range=[load, load], 
        goal_pressure_R_range=[0.3, 2.0],
        goal_pressure_L_range=[0.3, 2.0],
        max_episodes=100,
        use_combined_loss = True,
        use_step_loss = False,   
        # obs_noise = 0.01,
        # K_deform = 0.01,
        )
    load = str(load).replace('.', '_')

    env.reset()

    GL=env.goal_pressure_sequence_L
    GR=env.goal_pressure_sequence_R

    # Load the trained agent

    # Test
    # model_dir = "remote_models"
    # model_run = "1671860789"
    # model_step = "11000000"       

    model_dir = "models"
    model_run = "1673923036"
    model_step = "4000000"    

    model_path = f"{model_dir}/{model_run}/{model_step}"  # for var load experiment
    model = SAC.load(model_path, env=env, print_system_info=True)

    # Evaluate the agent
    print("Evaluating the agent with 100 episodes. Please wait")
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
    MEAN_REWARD.append(mean_reward)
    STD_REWARD.append(std_reward)
    reward = 0
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward}\n")

    # Enjoy trained agent
    episodes = 10
    for ep in range(episodes):
        obs = env.reset()
        VL, VR = env.pump.V_L/env.pump.Lchamber.V, env.pump.V_R/env.pump.Rchamber.V
        # env.goal_pressure_sequence_L=GL
        # env.goal_pressure_sequence_R=GR
        # env.render()
        # Set load and goal pressure
        # obs = env.set_load_and_goal_pressure(load=1.5, goal_pressure=1.85*P_0)
        
        # print('env.load', env.load, 'env.goal_pressure', env.goal_pressure/P_0)
        # print("Goal pressure: {p:.2f}".format(p=env.goal_pressure/P_0), '=', env.goal_pressure, 'Pa')

        P_L, P_R, P_R_G, P_L_G, R = [], [],[],[],[]
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
        while not done:


            env_filename ="./saved_figures/{}/{}/step_{:03d}.png".format(load, ep, env.action_counter) 
            tracking_filename = "./saved_figures/{}/{}/tracking_step_{:03d}.png".format(load, ep, env.action_counter)
            env_render = env.render(
                title="step: {}".format(env.action_counter), 
                # filename=env_filename,
                # time=1
            )

            P_R.append((env.pump.Rchamber.load_P-P_0)/1000)
            P_L.append((env.pump.Lchamber.load_P-P_0)/1000)
            R.append(reward)
            
            # # env.goal_pressure_sequence_L/1000
            # # env.goal_pressure_sequence_R/1000
            pickle.dump([P_L, P_R, P_R_G, P_L_G, R, VL, VR], open( "./scripts/save.p", "wb" ) )
            os.system("python ./scripts/plot_goals_two.py")
            # cv2.imshow('Goals vs achieved: Trained on sequential goals', cv2.imread('./scripts/file.png'))
            os.system("cp ./scripts/file.png {}".format(tracking_filename))
            
            A = env_render # cv2.imread(env_filename)
            A = A[50:420,:,:]
            B = cv2.imread(tracking_filename)
            C = np.concatenate((A,B), axis=0)
            cv2.imwrite(env_filename, C)

            # print("obs:", obs[-1])
            # print('env.load', env.load)
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            # print("Rchamber.P: {p:.3f} | ".format(p=env.pump.Rchamber.P/P_0),
            #       "Goal {p1} pressure {p2:.3f} | ".format(p1=env.goal_sequence_number, p2=env.goal_pressure/P_0),
            #       "Step reward: {p} | ".format(p=reward),
            #       "Error: {p:.3f} | ".format(
            #         p = (env.pump.Rchamber.P - env.goal_pressure)/env.goal_pressure
            #         # p = abs((env.pump.Rchamber.P - env.goal_pressure)/env.goal_pressure)
            #         )
            # )

        
        pickle.dump([P_L, P_R, P_R_G, P_L_G, R, VL, VR], open( "./scripts/save.p", "wb" ) )
        os.system("python ./scripts/plot_goals_two.py")
        cv2.imshow('Goals vs achieved: Trained on sequential goals', cv2.imread('./scripts/file.png'))
        cv2.waitKey(1)


        os.system("python ./scripts/plot_goals_two_square.py")
        os.system("cp ./scripts/save.p ./saved_figures/{}/save_{}.p".format(load, ep))
        os.system("cp ./scripts/file.png ./saved_figures/{}/tracking_results_ep_{}.png".format(load, ep))
        os.system("rm -rf ./saved_figures/*/*/tracking_step_*")
        
        os.system("ffmpeg -r 3 -i ./saved_figures/{p1}/{p2}/step_%03d.png -vcodec mpeg4 -y ./saved_figures/{p1}/tracking_results_ep_{p2}.mp4".format(p1=load, p2=ep))

        # cv2.waitKey(0)

pickle.dump([test_loads, MEAN_REWARD,STD_REWARD], open( "./scripts/load_test_results.p", "wb" ) )
os.system("python ./scripts/plot_load_test_results.py")
# cv2.imshow('Mean reward with different loads, evaluated with 1000 episodes', cv2.imread('./scripts/file.png'))
# cv2.waitKey(0)

os.system("cp ./scripts/file.png ./saved_figures/overall_results.png")