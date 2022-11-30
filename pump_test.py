from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, DDPG
from pump_env import PumpEnv
from pump_env_variable_load import PumpEnvVar
import numpy as np

P_0 = 1.01*1e5  # Pa


# Create environment
env = PumpEnvVar()  # with load
env.reset()

# Load the trained agent
model_dir = "models"
model_run = "1669720397"
model_step = "2000000"
# model_path = f"{models_dir}/1668239166/1600000"
model_path = f"{model_dir}/{model_run}/{model_step}"  # for var load experiment
model = DDPG.load(model_path, env=env, print_system_info=True)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward}")

# Enjoy trained agent
episodes = 1
for ep in range(episodes):
    env.reset()
    # Set load and goal pressure
    goal_pressure = np.random.random(size=None)*5 # random number between 0 to 1
    obs = env.set_load_and_goal_pressure(var_L=0.012, goal_pressure=goal_pressure*P_0) # goal_pressure=2.62*P_0
    # obs = env.set_load_and_goal_pressure(var_L=0.012, goal_pressure=2.62*P_0) # 

    print("Goal pressure: {p:.2f}".format(p=env.goal_pressure/P_0), '=', env.goal_pressure, 'Pa')
    # env.pump.graphics.add_text_to_image("Goal pressure: {p1:.2f} = {p2:.2f} pa".format(p1=env.goal_pressure/P_0, p2=env.goal_pressure), (400,90))

    # Start simulation
    env.render()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print("Rchamber.P: {p:.2f};".format(p=env.pump.Rchamber.P/P_0),
               done,
              "Error: {e:.3f}.".format(e=abs(env.pump.Rchamber.P - env.goal_pressure)/env.goal_pressure))
        env.render()
