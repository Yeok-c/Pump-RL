from stable_baselines3 import PPO, DDPG, TD3, SAC
from stable_baselines3.common.noise import NormalActionNoise
import os, sys, getopt, time
from curi_communication_udp import curi_communication_udp
from pump_realenv_variable_load_two_DRA import PumpRealEnvVar_Two
# from pump_env_variable_load_two_DRA import PumpEnvVar_Two
from scripts.get_args import get_args
from stable_baselines3.common.env_util import make_vec_env
from typing import Callable
from summary_writer import SummaryWriterCallback
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
import pickle
import numpy as np

def demo_dog(env, step_size=0.1):
    P_M_ =[]
    LCHAMBER_P_=[]
    RCHAMBER_P_=[]

    # Example action
    # (c) Max right, open right load valve (equalize load and chamber pressures)
    # [ 1, 0, 0, 0, 0, 1, 0,], 
    #   PM N  lL cL cR lR I  notation
    pm = 0
    def _left_to_right(title=''):        
        experiment_range = np.arange(-1, 1, step_size)
        # First loop: -0.08 to 0.08
        for pm in experiment_range:
            action = [pm, 1, 0, 0, 0, 0, 0,] # Open no valves, move to p_m 
            _,_,_,_ = env.step(action)
            env.render(1, title='{}, P_M={:.05f}'.format(title, env.pump.P_M))
            _print()
    
    def _right_to_left(title=''):
        experiment_range = -np.arange(-1, 1, step_size)
        # Second loop: 0.08 to -0.08
        for pm in experiment_range:
            action = [pm, 1, 0, 0, 0, 0, 0,] # Open no valves, move to p_m 
            _,_,_,_ = env.step(action)
            env.render(1, title='{}, P_M={:.05f}'.format(title, env.pump.P_M))
            _print()
    
    def _action(action, title=''):
        _,_,_,_ = env.step(action)
        env.render(1, title='{}, P_M={:.05f}'.format(title, env.pump.P_M))

    def _print():
        print("P_M, P_L, P_R | {: .05f} | {: 7.05f} | {: 7.05f}".format(
            env.pump.P_M, env.pump.pressure[0], env.pump.pressure[1]
        # self.env.Lchamber.P =   self.env.pressure[0]
        # self.env.Rchamber.P =   self.env.pressure[1]
        ))
        P_M_.append(env.pump.P_M)
        LCHAMBER_P_.append(env.pump.pressure[0])
        RCHAMBER_P_.append(env.pump.pressure[1])
    

    # # soft reset
    # self.env.pump.set_valves([1,1,1,1,1])
    # self.env.pump.set_position(0)
    # self.env.pump.set_valves([0,0,0,0,0])

    # Pumping air into the right
    #         PM N  lL cL cR lR I  notation


    def _charge_r(both_valves=False):
        # _left_to_right(title='1st loop pumping air into the right')
        # _action([-1, 0, 0, 0, 1, 0, 0,], title='Move left, open right valve') # Move left, open right valve
        # _action([ 1, 0, 0, 1, 0, 0, 0,], title='Move right, open left valve')
        # _action([-1, 0, 0, 0, 0, 0, 1,], title='Move left, open inner valve')
        # _action([ 1, 0, 0, 0, 0, 1, 0,], title='Move right, open right load valve') 

        env.pump.set_valves([0, 0, 1, 1, 0]) # set loads to open
        _action([-1, 0, 0, 0, 1, 0, 0,], title='Move left, open right valve') # Move left, open right valve
        _action([ 1, 0, 0, 0, 0, 1, 0,], title='Move right, open right load valve') 
        # env.pump.set_valves([0, 0, 0, 1, 0]) # set loads to open

#                     PM N  lL cL cR lR I  notation
        for i in range(3):
            _action([-1, 0, 0, 0, 0, 0, 1,], title='Move left, open right valve') # Move left, open right valve
            _action([ 1, 0, 0, 1, 0, 0, 0,], title='Move right, open inner valve')


        # _action([-1, 0, 0, 0, 1, 0, 0,], title='Move left, open right valve') # Move left, open right valve
        # _action([ 1, 0, 0, 1, 0, 0, 0,], title='Move right, open left valve')
        # _action([-1, 0, 0, 0, 0, 0, 1,], title='Move left, open inner valve')
        # _action([ 1, 1, 0, 0, 0, 0, 0,], title='Move right, open right load valve') 
        if both_valves == False:
            env.pump.set_valves([0, 0, 0, 1, 0]) # set loads to open
            env.pump.set_valves([0, 0, 0, 1, 0]) # set loads to open
        else:
    #         PM N              lL cL cR lR I  notation

            # env.pump.set_valves([1, 0, 0, 1, 0]) # set loads to open
            # env.pump.set_valves([0, 0, 0, 1, 0]) # set loads to open
            env.pump.set_valves([1, 1, 0, 1, 0]) # set loads to open
            env.pump.set_valves([1, 1, 0, 1, 0]) # set loads to open
            env.pump.set_valves([1, 1, 0, 1, 0]) # set loads to open
#                 PM N  lL cL cR lR I  notation
            # _action([-1, 0, 0, 1, 0, 0, 0,], title='Move left, open right valve') # Move left, open right valve
            # _action([ 1, 1, 0, 0, 0, 0, 0,], title='Move right, open inner valve')
            # env.pump.set_valves([1, 0, 0, 0, 0]) # set loads to open
            # env.pump.set_valves([1, 0, 0, 0, 0]) # set loads to open
            
        # _action([-1, 0, 0, 0, 1, 0, 0,], title='Move left, open right valve') # Move left, open right valve
        # _action([ 1, 0, 0, 1, 0, 0, 0,], title='Move right, open left valve')
        # _action([-1, 0, 0, 0, 0, 0, 1,], title='Move left, open inner valve')
        # # _left_to_right(title='2nd loop pumping air into the right')
        # _action([ 1, 0, 0, 0, 0, 1, 0,], title='Move right, open right load valve') 

    # Pumping air into the left
    #         PM N  lL cL cR lR I  notation

    # # Eq all
    # env.pump.set_valves([1,1,1,1,1])
    # env.pump.set_valves([0,0,0,0,0])
    def _charge_l(both_valves=False):
        # _right_to_left(title='1st loop pumping air into the left')
        # _action([ 1, 0, 0, 1, 0, 0, 0,], title='Move right, open left valve') 
        # _action([-1, 0, 0, 0, 1, 0, 0,], title='Move left, open right valve')
        # _action([ 1, 0, 0, 0, 0, 0, 1,], title='Move right, open inner valve')
        # _action([-1, 0, 1, 0, 0, 0, 0,], title='Move right, open right load valve') 

#                     PM N  lL cL cR lR I  notation
        env.pump.set_valves([1, 1, 0, 0, 0]) # set loads to open
        _action([ 1, 0, 0, 1, 0, 0, 0,], title='Move right, open left valve') 
        _action([-1, 0, 1, 0, 0, 0, 0,], title='Move left, open right load valve') 
        # env.pump.set_valves([1, 0, 0, 0, 0]) # set loads to open

        for i in range(3):
    #                 PM N  lL cL cR lR I  notation
            _action([ 1, 0, 0, 0, 0, 0, 1,], title='Move right, open left valve') 
            _action([-1, 0, 0, 0, 1, 0, 0,], title='Move left, open inner valve')

        if both_valves == False:
            env.pump.set_valves([1, 0, 0, 0, 0]) # set loads to open
            env.pump.set_valves([1, 0, 0, 0, 0]) # set loads to open
        else:

    #         PM N              lL cL cR lR I  notation
            # env.pump.set_valves([1, 0, 0, 0, 0]) # set loads to open
            env.pump.set_valves([1, 0, 1, 1, 0]) # set loads to open
            env.pump.set_valves([1, 0, 1, 1, 0]) # set loads to open
            env.pump.set_valves([1, 0, 1, 1, 0]) # set loads to open
    #                 PM N  lL cL cR lR I  notation
            # _action([ 1, 0, 0, 0, 1, 0, 0,], title='Move right, open left valve') 
            # _action([-1, 1, 0, 0, 0, 0, 0,], title='Move left, open inner valve')
            # env.pump.set_valves([0, 0, 0, 1, 0]) # set loads to open
            # env.pump.set_valves([0, 0, 0, 1, 0]) # set loads to open

        # _action([ 1, 0, 0, 1, 0, 0, 0,], title='Move right, open left valve') 
        # _action([-1, 0, 0, 0, 1, 0, 0,], title='Move left, open right valve')
        # _action([ 1, 0, 0, 0, 0, 0, 1,], title='Move right, open inner valve')
        # # _right_to_left(title='2nd loop pumping air into the left')
        # _action([-1, 0, 1, 0, 0, 0, 0,], title='Move right, open right load valve') 
        

    # Eq all
    # env.pump.set_valves([1,1,1,1,1])
    # env.pump.set_valves([0,0,0,0,0])

    _charge_l()
    # _charge_l()
    _charge_r()
    # _charge_r()
    _action([ 0, 1, 0, 0, 0, 0, 0,], title='Reset to initial pos')

    # env.pump.set_valves([1, 0, 0, 1, 0]) # set loads to open

    for i in range(20):
        # env.pump.set_position(0.008)
        # env.pump.set_position(-0.008)
    #         PM N  lL cL cR lR I  notation
        print("Charging L")
        _charge_l(both_valves=True)
        print("Charging R")
        _charge_r(both_valves=True)
        
        #         PM N  lL cL cR lR I  notation
        # _action([ 1, 0, 0, 0, 0, 1, 0,], title='Move right, open right load valve') 
        # _action([-1, 0, 0, 0, 0, 1, 0,], title='Move left, open right load valve') 
        # _action([-1, 0, 1, 0, 0, 0, 0,], title='Move left, open right load valve') 
        # _action([ 1, 0, 1, 0, 0, 0, 0,], title='Move right, open right load valve') 
        # env.pump.set_valves([1,1,1,1,1])

    # Eq all        
    env.pump.set_valves([1,1,1,1,1])
    env.pump.set_position(0)
    env.pump.set_valves([0,0,0,0,0])


    pickle.dump([P_M_, LCHAMBER_P_, RCHAMBER_P_], open( "./scripts/chamber_experiments_real_009_updated_motor_r.p", "wb" ) )


if "__main__" == __name__:

    udp = curi_communication_udp("127.0.0.1", 13331, "127.0.0.1", 13332)
    udp.open()
    print("Open udp")

    env = PumpRealEnvVar_Two(
        udp = udp,
        load_range=[0, 2.0], 
        goal_pressure_R_range=[0.5, 1.7],
        goal_pressure_L_range=[0.5, 1.7],
        max_episodes=10000,
        use_combined_loss = True,
        use_step_loss = False,
        )

    demo_dog(env)