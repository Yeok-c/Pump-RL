from stable_baselines3 import PPO, DDPG, TD3
from custom_policy import TD3Policy_embedding # right click to see
import os
import time
# from pump_env import PumpEnv
# from pump_env_variable_load import PumpEnvVar
# from pump_env_variable_load_two import PumpEnvVar_Two 
# from pump_env_variable_load_changing_goal_fixed_length import PumpEnvVar_Sequence
from pump_env_variable_load_two_changing_goal_fixed_length import PumpEnvVar_Two
from stable_baselines3.common.env_util import make_vec_env


# Create dirs
models_dir = f"models/{int(time.time())}"
logs_dir = f"../logs/{int(time.time())}"
logname = "TD3 Regular Tasks, Known Variable Load 1_0-1_1"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Environment
# env = PumpEnvVar(load_range=[0.0,2.0], goal_pressure_range=[1.1, 2.0])
# env = PumpEnvVar_Sequence(
#     load_range=[0.0,2.0], 
#     goal_pressure_range=[1.1, 2.0],
#     goal_sequence_length=100,
#     episode_length=100
#     )

env = PumpEnvVar_Two(
    load_range=[1, 1.1], 
    goal_pressure_R_range=[1.01, 1.9],
    goal_pressure_L_range=[0.4, 0.99],
    max_episodes=20,
    use_combined_loss = True,
    use_step_loss = False
    )

# env = make_vec_env(lambda: env, n_envs=4)  # Multi-process (This behaves like batchsize)
env.reset()

# policy_kwargs = dict(
#     # activation_fn=th.nn.ReLU,
#     net_arch=[200, 200]
#     )

# Model
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir)
model = TD3(
    TD3Policy_embedding,
    # "MlpPolicy",
    env, verbose=1, 
    tensorboard_log=logs_dir, 
    learning_rate=0.001,
    # policy_kwargs=policy_kwargs,
)

# # Load the trained agent
# model_dir = "remote_models"
# model_run = "1670811561"
# model_step = "1600000"         
# model_path = f"{model_dir}/{model_run}/{model_step}"  # for var load experiment
# model = DDPG.load(model_path, env=env, print_system_info=True, tensorboard_log=logs_dir) 


# Train and save every TIMESTEPS steps
TIMESTEPS = 500000
for i in range(1,int(10*1000000/TIMESTEPS)):
    # Turn off "reset_num_timesteps" so that the learning won't stop
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=logname, progress_bar=True)
    
    model_filepath = f"{models_dir}/{TIMESTEPS*i}"
    model.save(model_filepath)
env.close()

# zip, send
# os.system(f"zip -r final_model_{logname}.zip ./{model_filepath}/")
desired_filename = f"{model_filepath}.zip".replace('/', '_')
newpath = f"./{logname}_{desired_filename}"
os.system(f"mv ./{model_filepath}.zip {newpath}")
os.system(f"expect send_.exp {newpath}")
# Note on tensorboard
# Command line: tensorboard --logdir=logs
# Then copy paste the link to your browser