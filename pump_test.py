from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, DDPG
from pump_env import PumpEnv
from pump_env_variable_load import PumpEnvVar

P_0 = 1.01*1e5  # Pa


# Create environment
env = PumpEnvVar()  # with load  
env.reset()

# Load the trained agent
model_dir = "models"
model_run = "1670221230" #"1670211944"
model_step = "690000"
model_path = f"{model_dir}/{model_run}/{model_step}"  # for var load experiment
model = PPO.load(model_path, env=env, print_system_info=True)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward}\n")



# Enjoy trained agent
episodes = 1
for ep in range(episodes):   
    env.reset()
    # Set load and goal pressure
    # obs = env.set_load_and_goal_pressure(load=env.get_random_load(), goal_pressure=env.get_random_goal())
    obs = env.set_load_and_goal_pressure(load=1.5, goal_pressure=1.85*P_0)
    print('env.load', env.load, 'env.goal_pressure', env.goal_pressure/P_0)
    print("Goal pressure: {p:.2f}".format(p=env.goal_pressure/P_0), '=', env.goal_pressure, 'Pa')
    # Start simulation
    env.render()
    done = False
    while not done:
        print("obs:", obs[-1])
        print('env.load', env.load)
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print("Rchamber.P: {p:.2f};".format(p=env.pump.Rchamber.P/P_0),
               done,
              "Error: {e:.3f}.".format(e=abs(env.pump.Rchamber.P - env.goal_pressure)/env.goal_pressure))
        env.render()
