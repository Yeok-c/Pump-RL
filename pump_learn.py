from stable_baselines3 import PPO, DDPG
import os
import time
from pump_env import PumpEnv
from pump_env_variable_load import PumpEnvVar
from stable_baselines3.common.env_util import make_vec_env


# Create dirs
models_dir = f"models/{int(time.time())}"
logs_dir = f"logs/{int(time.time())}"
logname = "DDPG_Vanilla"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Environment
env = PumpEnvVar(load_range=[0.0,2.0], goal_pressure_range=[1.1, 2.0])
# env = make_vec_env(lambda: env, n_envs=4)  # Multi-process (This behaves like batchsize)
env.reset()

# Model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir)
# model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir)

# Load the trained agent
# model_dir = "models"
# model_run = "1670394469"
# model_step = "2900000"
# model_path = f"{model_dir}/{model_run}/{model_step}"  # for var load experiment
# model = PPO.load(model_path, env=env, print_system_info=True)

# Train and save every TIMESTEPS steps
TIMESTEPS = 10000
for i in range(1,int(20000/TIMESTEPS)):
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