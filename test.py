import torch
# import numpy as np
import math
import scipy.signal as signal
import random

P_0 = 1.01*1e5  # Pa
pi = torch.tensor(math.pi)

class Env():
    def __init__(self):
        self.max_episodes=100
        self.future_frames=3
        

    def generate_moving_goal_pressure(self, amp_range, period=2*pi, phase=0):
    # def generate_moving_goal_pressure(self, range, period=2*np.pi, K_amp_noise=0, K_phase_noise=0):
        # assert K_amp_noise >= 0 and K_amp_noise <= 1
        # assert K_phase_noise >= 0 and K_phase_noise <= 1
        x = torch.arange(0, self.max_episodes+self.future_frames, 1)
        # phase_noise = np.random.uniform(0,2*np.pi, (1,)) 
        # amp_noise = np.random.uniform(-1,1, (self.max_episodes,))
        
        wave_style= torch.randint(3, (1,)) # np.random.randint(3)
        # print(wave_style)
        if wave_style==0:
            y = torch.sin(x/period + phase)
        if wave_style==1:
            y = torch.tensor(signal.square(x/period)) 
        if wave_style==2:
            y = torch.tensor(signal.sawtooth(x/period))
        # y = 0.5*((1-K_amp_noise)*np.sin(x/period + K_phase_noise*phase_noise) + K_amp_noise*amp_noise)
        # result will always be between -0.5 to -0.5, range of 1.0

        float_offset = 0.000005
        amp_range=torch.squeeze(amp_range)
        # assert amp_range[1] > amp_range[0]
        # amp_range=torch.sort(amp_range)
        amp_range_ = (amp_range[1] - amp_range[0])/2
        mean = amp_range[0] + amp_range_
        y = mean + (amp_range_-float_offset)*y # mean + scaled y
        assert torch.min(y) >= amp_range[0] # min value should still be larger than range
        assert torch.max(y) <= amp_range[1] # max value should still be smaller than range
        return y

    def generate_composite_moving_goal_pressure(self, amp_range, n=3):
        def _generate_goal_pressure_sequence():
            amp_range_this, _ = torch.sort(
                    torch.FloatTensor(1, 2).uniform_(amp_range[0],amp_range[1]))
            return self.generate_moving_goal_pressure(
            # amp_range= th.sort(np.random.uniform(amp_range[0],amp_range[1], 2)),
            amp_range=amp_range_this,
                # np.random.uniform(amp_range[0],amp_range[1], 2)),
            phase = torch.FloatTensor(1, 1).uniform_(0,2*pi), # random.uniform(0,2*np.pi),
            period = torch.FloatTensor(1, 1).uniform_(1*pi, 5*pi), #random.uniform(1*np.pi, 5*np.pi)
            ) * P_0

        goal_pressure_sequence = _generate_goal_pressure_sequence()
        for i in range(1,n):
            goal_pressure_sequence = torch.column_stack((goal_pressure_sequence, _generate_goal_pressure_sequence()))

        goal_pressure_sequence = torch.mean(goal_pressure_sequence, 0)

        return goal_pressure_sequence


env = Env()
A = torch.FloatTensor(1, 1).uniform_(-1, 1)
print(env.generate_moving_goal_pressure(torch.tensor([0,1])))


