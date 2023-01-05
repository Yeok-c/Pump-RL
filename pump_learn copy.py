from stable_baselines3 import PPO, DDPG, TD3, SAC
import os, sys, getopt, time
from pump_env_variable_load_two_DRA import PumpEnvVar_Two
from scripts.get_args import get_args
from stable_baselines3.common.env_util import make_vec_env
from typing import Callable
from summary_writer import SummaryWriterCallback

# Usage
# CUDA_VISIBLE_DEVICES=0 python pump_learn.py --noise 0.05 --dra_schedule 0 --goal_range_low 0.3 --goal_range_high 3.0 --load_vol_range_low 0.0 --load_vol_range_high 2.0 --gamma 0.92 --timesteps 50
# if no arguments are set then the following defaults are used

def get_args(argv):
    noise = 0.05
    dra_schedule = 0 # 0 constant
    goal_range_low = 0.3
    goal_range_high = 3.0
    load_vol_range_low = 0
    load_vol_range_high = 2.0
    gamma = 0.92
    timesteps = 50*1000000

    opts, args = getopt.getopt(argv,"hi:o:",[
    "noise=","dra_schedule=", 
    "goal_range_low=", "goal_range_high=", 
    "load_vol_range_low=", "load_vol_range_high=",
    "gamma=", "timesteps="])


    for opt, arg in opts:
        if opt == '-h':
            print ('Example: CUDA_VISIBLE_DEVICES=0 python pump_learn.py --noise 0.05 --dra_schedule 0 --goal_range_low 0.3 --goal_range_high 3.0 --load_vol_range_low 0.0 --load_vol_range_high 2.0 --gamma 0.92 --timesteps 50000000')
            sys.exit()

        elif opt in ("--noise"):
            noise = float(arg)
        
        elif opt in ("--dra_schedule"):
            dra_schedule = int(arg)

        elif opt in ("--goal_range_low"):
            goal_range_low = float(arg)

        elif opt in ("--goal_range_high"):
            goal_range_high = float(arg)

        elif opt in ("--load_vol_range_low"):
            load_vol_range_low = float(arg)

        elif opt in ("--load_vol_range_high"):
            load_vol_range_high = float(arg)

        elif opt in ("--gamma"):
            gamma = float(arg)

        elif opt in ("--timesteps"):
            timesteps = int(arg)*1000000

    return noise, dra_schedule, goal_range_low, goal_range_high, load_vol_range_low, load_vol_range_high, gamma, timesteps

if __name__ == "__main__":
    noise, dra_schedule, goal_pressure_L, goal_pressure_H, \
        load_range_L, load_range_H, gamma, timesteps = get_args(sys.argv[1:])

    # Create dirs
    models_dir = f"models/{int(time.time())}"
    logs_dir = f"../logs/{int(time.time())}"
    # logname="SACSHORT"
    logname = "SAC Noise{}_Schedule{}_Goalrange{}-{}_Loadrange{}-{}_Gamma{}_Steps{}M".format(
        noise, dra_schedule, goal_pressure_L, goal_pressure_H, load_range_L, load_range_H,
        gamma, timesteps/1000000
    )

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    env = PumpEnvVar_Two(
        load_range=[load_range_L, load_range_H], 
        goal_pressure_R_range=[goal_pressure_L, goal_pressure_H],
        goal_pressure_L_range=[goal_pressure_L, goal_pressure_H],
        max_episodes=100,
        use_combined_loss = True,
        use_step_loss = False,
        )
    env.reset()



    # Load the trained agent
    # model_dir = "models"
    # model_run = "1671515701"
    # model_step = "9500000"     
    # env = make_vec_env(lambda: env, n_envs=10)  # Multi-process (This behaves like batchsize)    
    # model_filepath = f"remote_models/1671515701/9500000"  # for var load experiment
    # model = SAC.load(
    #     model_filepath, env=env, print_system_info=True, tensorboard_log=logs_dir,
    #     # buffer_size=1000000, 
    #     gamma=0.95,
    #     batch_size=2048,
    #     ) 

    policy_kwargs = dict(
        # activation_fn=th.nn.ReLU,
        net_arch=[64, 64]
        )

    # Model
    model = SAC(
        # TD3Policy_embedding,
        "MlpPolicy",
        env, verbose=1, 
        tensorboard_log=logs_dir, 
        
        gamma=0.92,
        # action_noise=0.01,
        batch_size=2048,
        # learning_rate=linear_schedule(0.001),
        policy_kwargs=policy_kwargs,
    )

    SCHED_TIMESTEPS = 100000
    SAVE_TIMESTEPS = SCHED_TIMESTEPS*10
    TOTAL_TIMESTEPS = timesteps # 50*1000000

    # try:
    for i in range(0, TOTAL_TIMESTEPS, SAVE_TIMESTEPS):
        for j in range(0, SAVE_TIMESTEPS, SCHED_TIMESTEPS):
            progress = (i+j)/TOTAL_TIMESTEPS
            if dra_schedule == 1:
                training_noise = progress*noise
            elif dra_schedule == 0: # no schedule, constant
                training_noise = noise
            print("Current progress:{}, noise: {}".format(progress, noise))

            env = PumpEnvVar_Two(
                load_range=[load_range_L, load_range_H], 
                goal_pressure_R_range=[goal_pressure_L, goal_pressure_H],
                goal_pressure_L_range=[goal_pressure_L, goal_pressure_H],
                max_episodes=100,
                use_combined_loss = True,
                use_step_loss = False,
                obs_noise = training_noise,
                K_deform = training_noise,
                )

            # env = make_vec_env(lambda: env, n_envs=10)  # Multi-process (This behaves like batchsize)    
            model.set_env(env)
            model.learn(total_timesteps=SCHED_TIMESTEPS, reset_num_timesteps=False, 
            tb_log_name=logname, callback=SummaryWriterCallback(), progress_bar=True)    
            model.save(f"{models_dir}/{i}")

    print("Training complete: {}".format(TOTAL_TIMESTEPS))
    model.save(f"{models_dir}/{TOTAL_TIMESTEPS}")

    # except:
    #     print("disk full")
    #     pass

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