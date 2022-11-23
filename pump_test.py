from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from pump_env import PumpEnv
from pump_env_variable_load import PumpEnvVar

P_0 = 1.01*1e5  # Pa


# Create environment
env = PumpEnvVar()  # with load
env.reset()

# Load the trained agent
model_dir = "models"
model_run = "1669172947"
model_step = "900000"
# model_path = f"{models_dir}/1668239166/1600000"
model_path = f"{model_dir}/{model_run}/{model_step}"  # for var load experiment
model = PPO.load(model_path, env=env, print_system_info=True)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward}")

# Enjoy trained agent
episodes = 1
for ep in range(episodes):
    env.reset()
    # Set load and goal pressure
    obs = env.set_load_and_goal_pressure(var_L=0.012, goal_pressure=2.62*P_0)
    print("Goal pressure: {p:.2f}".format(p=env.goal_pressure/P_0), '=', env.goal_pressure, 'Pa')
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
