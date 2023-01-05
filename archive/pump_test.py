from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, DDPG, TD3
from pump_env import PumpEnv
from pump_env_variable_load import PumpEnvVar
from pump_env_variable_load_two import PumpEnvVar_Two 
from pump_env_variable_load_changing_goal_fixed_length import PumpEnvVar_Sequence
import pickle
import os
import cv2

P_0 = 1.01*1e5  # Pa


# Create environment
env = PumpEnvVar_Sequence(
    load_range=[0.0,2.0], 
    goal_pressure_range=[1.1, 2.0],
    goal_sequence_length=100,
    episode_length=100
    )
env.reset()
goal_pressure_sequence = env.goal_pressure_sequence 

# Best agent: 
# model_run = "1670399531"
# model_step = "2500000"


# Load the trained agent
# model_dir = "models"
# Root
# model_run = "1670406961"
# model_step = "6200000"       

# Non sequence non reset
# model_run = "1670555494"
# model_step = "500000"       

# Best non-combined loss Sequence
# model_run = "1670560788"
# model_step = "200000"       

# Best combined loss sequence
# model_dir = "remote_models"
# model_run = "1670827566"
# model_step = "6500000"       

# Smoothest model
model_dir = "remote_models"
model_run = "1670830144"
model_step = "7900000"       

# Lowest loss model
# model_dir = "remote_models"
# model_run = "1670830144"
# model_step = "9100000"       

# for model_ep in range(100000, 9400000, 100000):
    # model_dir = "remote_models"
    # model_run = "1670830144"
    # model_step =  str(model_ep)      

model_path = f"{model_dir}/{model_run}/{model_step}"  # for var load experiment
model = TD3.load(model_path, env=env, print_system_info=True)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward}\n")

# Enjoy trained agent
episodes = 30
for ep in range(episodes):
    obs = env.reset()
    # env.goal_pressure_sequence=goal_pressure_sequence
    # Set load and goal pressure
    # obs = env.set_load_and_goal_pressure(load=1.5, goal_pressure=1.85*P_0)
    
    # print('env.load', env.load, 'env.goal_pressure', env.goal_pressure/P_0)
    # print("Goal pressure: {p:.2f}".format(p=env.goal_pressure/P_0), '=', env.goal_pressure, 'Pa')

    P=[]
    P_G=[]
    R=[]
    # Start simulation
    # env.render()
    done = False
    while not done:
        # print("obs:", obs[-1])
        # print('env.load', env.load)
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.print_step_info()
        # env.render()

        P.append(env.pump.Rchamber.P/P_0)
        P_G.append(env.goal_pressure/P_0) 
        R.append(reward)
        
    
    pickle.dump([P, P_G, R], open( "save.p", "wb" ) )
    os.system("python plot_goals.py")
    cv2.imshow('Goals vs achieved, no pretraining', cv2.imread('file.png'))
    os.system("cp file.png ./saved_figures/no_pretraining_combined_loss_episode_{}.png".format(ep))
    # cv2.waitKey(0)