from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from pump_env import PumpEnv


P_0 = 1.01*1e5  # Pa


# Create environment
env = PumpEnv(goal_pressure_range=[1.1, 4.0])
env.reset()


# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
models_dir = "models"
model_path = f"{models_dir}/1668239166/1600000"
model = PPO.load(model_path, env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward}")

# Enjoy trained agent
episodes = 1
for ep in range(episodes):
    obs = env.reset()
    # Set goal pressure
    env.set_goal_pressure(2.6*P_0)
    print("Goal pressure: {p:.2f}".format(p=env.goal_pressure/P_0), '=', env.goal_pressure, 'Pa')
    env.render()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print("Rchamber.P: {p:.2f};".format(p=env.pump.Rchamber.P/P_0),
               done,
              "Error: {e:.3f}.".format(e=abs(env.pump.Rchamber.P - env.goal_pressure)/env.goal_pressure)
            #   ,'abs(', env.pump.Rchamber.P,'-', env.goal_pressure, ')', '/', env.goal_pressure, '='
            #   ,abs(env.pump.Rchamber.P - env.goal_pressure)/env.goal_pressure
              )
        env.render()
