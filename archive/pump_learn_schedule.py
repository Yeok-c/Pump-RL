from stable_baselines3 import PPO, DDPG, TD3, SAC
from custom_policy import TD3Policy_embedding # right click to see
import os
import time
# from pump_env import PumpEnv
# from pump_env_variable_load import PumpEnvVar
# from pump_env_variable_load_two import PumpEnvVar_Two 
# from pump_env_variable_load_changing_goal_fixed_length import PumpEnvVar_Sequence
# from pump_env_variable_load_two_changing_goal_fixed_length import PumpEnvVar_Two
from pump_env_variable_load_two_DRA import PumpEnvVar_Two
from stable_baselines3.common.env_util import make_vec_env
from typing import Callable
import torch as th

# Create dirs
models_dir = f"models/{int(time.time())}"
logs_dir = f"../logs/{int(time.time())}"
logname = "SAC Regular Tasks, DRA, Noise, SameLRGoals"

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

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

env = PumpEnvVar_Two(
    load_range=[0.01, 2.0], 
    goal_pressure_R_range=[0.3, 2.0],
    goal_pressure_L_range=[0.3, 2.0],
    max_episodes=100,
    use_combined_loss = True,
    use_step_loss = False,
    )

# env = make_vec_env(lambda: env, n_envs=4)  # Multi-process (This behaves like batchsize)
env.reset()

# policy_kwargs = dict(
#     # activation_fn=th.nn.ReLU,
#     net_arch=[64, 64]
#     )

# Model
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir)
# model = SAC(
#     # TD3Policy_embedding,
#     "MlpPolicy",
#     env, verbose=1, 
#     tensorboard_log=logs_dir, 
#     # learning_rate=linear_schedule(0.001),
#     policy_kwargs=policy_kwargs,
# )
# Load the trained agent
model_dir = "models"
model_run = "1671515701"
model_step = "9500000"         
model_filepath = f"{model_dir}/{model_run}/{model_step}"  # for var load experiment
model = SAC.load(model_filepath, env=env, print_system_info=True, tensorboard_log=logs_dir) 


SCHED_TIMESTEPS = 50000
SAVE_TIMESTEPS = SCHED_TIMESTEPS*10
TOTAL_TIMESTEPS = 20*1000000

for i in range(0, TOTAL_TIMESTEPS, SAVE_TIMESTEPS):
    for j in range(0, SAVE_TIMESTEPS, SCHED_TIMESTEPS):
        progress = (i+j)/TOTAL_TIMESTEPS
        noise = progress*0.1
        print("Current progress:{}, noise: {}".format(progress, noise))
        env = PumpEnvVar_Two(
            load_range=[0.01, 2.0], 
            goal_pressure_R_range=[0.3, 2.0],
            goal_pressure_L_range=[0.3, 2.0],
            max_episodes=100,
            use_combined_loss = True,
            use_step_loss = False,
            obs_noise = 0.0,
            K_deform = noise,
            )

        # Load the trained agent
        # model = SAC.load(model_filepath, env=env, print_system_info=True, tensorboard_log=logs_dir, verbose=0) 
        # os.system(f"rm {model_filepath}.zip")
        model.set_env(env)
        model.learn(total_timesteps=SCHED_TIMESTEPS, reset_num_timesteps=False, tb_log_name=logname, progress_bar=False)    
        # model_filepath = f"{models_dir}/{model_run}/{j}"
        # model.save(model_filepath)
    model.save(f"{models_dir}/{model_run}/{i}")

print("Training complete: {}".format(TOTAL_TIMESTEPS))
model.save(f"{models_dir}/{model_run}/{i}")
env.close()

# # Train and save every TIMESTEPS steps
# TIMESTEPS = 500000
# for i in range(int(10*1000000/TIMESTEPS)):
#     # Turn off "reset_num_timesteps" so that the learning won't stop
#     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=logname, progress_bar=True)    
#     model_filepath = f"{models_dir}/{TIMESTEPS*i}"
#     model.save(model_filepath)


# # zip, send
# # os.system(f"zip -r final_model_{logname}.zip ./{model_filepath}/")
# desired_filename = f"{model_filepath}.zip".replace('/', '_')
# newpath = f"./{logname}_{desired_filename}"
# os.system(f"mv ./{model_filepath}.zip {newpath}")
# os.system(f"expect send_.exp {newpath}")
# # Note on tensorboard
# # Command line: tensorboard --logdir=logs
# # Then copy paste the link to your browser