# The Simulator for the hybrid pump
import numpy
import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
from resources.graphics import Graphics


P_0 = 1.01*1e5  # Pa


class chamber:
    def __init__(self, L0, P0):
        # self.radius = 0.02  # Radius of chamber
        self.a = 0.02  # The short edge of chamber cross surface
        self.b = 0.03  # The long edge of chamber cross surface
        self.n = 6   # The number of element size
        self.L = L0  # Length of chamber
        self.P = P0  # Pressure of chamber
        self.V = self.calculate_V()  # Volume of chamber
        self.P_prev = P0  # Pressure of chamber in previous step
        self.V_prev = self.V   # Volume of chamber in previous step

        return
    def calculate_V(self, L=None):
        if L is None:
            L = self.L
        # Calculate the volume of chamber (V) by the length (L)
        # compensate = - 0.002 * math.pi * self.radius * self.radius
        # V = L * math.pi * self.radius * self.radius

        # Calculate the volume of chamber (V) by the length (L)
        compensate = -2.5E-06
        h = L / self.n
        V_seg = (5*self.b*self.b/6 + self.a*self.b/3 - self.a*self.a/6) * h
        V = 2 * self.n * V_seg + compensate
        return V
    def calculate_P(self):
        # Calculate the pressure of chamber (P) by the volume (V)
        return self.P_prev * self.V_prev / self.V
    def change_length(self, dL):
        self.V_prev = self.V
        self.P_prev = self.P
        if self.L + dL < 0:
            print("Length error", "prev L:", self.L, ".And the dL is", dL, 'resulting L:', self.L + dL)
        self.L = self.L + dL
        self.V = self.calculate_V()  # update V
        self.P = self.calculate_P()  # update P
        return

# \/__|——————|__\/__|——————|__\/
# /\  |——————|  /\  |——————|  /\
# IO0  VL, PL   IO1  VR, PR   IO2
class hybrid_pump:
    def __init__(self, L_L, L_R):
        self.P_M = 0  # motor
        self.P_M_L_Limitation = -0.05
        self.P_M_R_Limitation = +0.05
        self.Lchamber = chamber(L_L, P_0)
        self.Rchamber = chamber(L_R, P_0)
        self.valve = [0,0,0]  # 0: close, 1: open  # L, R, inner
        self.Lchamber_V_max = self.Lchamber.calculate_V(L_L + abs(self.P_M_R_Limitation))
        self.Rchamber_V_max = self.Rchamber.calculate_V(L_R + abs(self.P_M_L_Limitation))
        self.Lchamber_V_min = self.Lchamber.calculate_V(L_L - abs(self.P_M_L_Limitation))
        self.Rchamber_V_min = self.Rchamber.calculate_V(L_R - abs(self.P_M_R_Limitation))
        # self.graphics = Graphics() # not sure why cannot initialize together with hybrid pump :o


    def render(self, time=0):
        # cv2.rectangle(self.img,(100,100),(100+int(0.2*MM_2_PX),100-10),(0,255,0),3)  # pump
        # cv2.circle(self.img,(int(200+self.P_M*MM_2_PX),100-5),8,(255,0,0),-1)  # motor
        # chamber length = int(0.2*MM_2_PX)
        # motor position self.P_M * MM_2_PX
        
        # if self.valve[0] == 0:
        #     cv2.circle(self.img,(90,100-5),3,(0,0,255),-1)  # IO0
        # if self.valve[1] == 0:
        #     cv2.circle(self.img,(310,100-5),3,(0,0,255),-1)   # IO1
        # if self.valve[2] == 0:
        #     cv2.circle(self.img,(200,100+10),3,(0,0,255),-1)  # IO2
        
        # cv2.putText(self.img,'Left P: \n {p:.2f}'.format(p=self.Lchamber.P/P_0),(250,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0),1,cv2.LINE_AA)
        # cv2.putText(self.img,'Right P: \n {p:.2f}'.format(p=self.Rchamber.P/P_0),(450,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0),1,cv2.LINE_AA)

        self.graphics = Graphics() # not sure why cannot initialize together with hybrid pump :o
        MM_2_PX = 1000
        self.graphics.render_valve_and_motor(self.valve, self.P_M*MM_2_PX)
        self.graphics.add_text_to_image('Left P: \n {p:.2f}'.format(p=self.Lchamber.P/P_0), (270,90))
        self.graphics.add_text_to_image('Right P: \n {p:.2f}'.format(p=self.Rchamber.P/P_0), (460,90))
        cv2.imshow('a',self.graphics.display_image)
        cv2.waitKey(time)

    def open_L_valve(self):
        self.valve[0] = 1
        self.Lchamber.P = P_0
        return
    def close_L_valve(self):
        self.valve[0] = 0
        return
    def open_R_valve(self):
        self.valve[1] = 1
        self.Rchamber.P = P_0
        return
    def close_R_valve(self):
        self.valve[1] = 0
        return
    def open_inner_valve(self):
        self.valve[2] = 1
        P_new = (self.Lchamber.P * self.Lchamber.V + self.Rchamber.P * self.Rchamber.V) / (self.Lchamber.V + self.Rchamber.V)
        self.Lchamber.P = P_new
        self.Rchamber.P = P_new
        return
    def close_inner_valve(self):
        self.valve[2] = 0
        return
    def move_motor_to_L(self, dL):
        if dL < 0:
            print('dL < 0')
            return
        if self.P_M - dL < self.P_M_L_Limitation:
            dL = self.P_M - self.P_M_L_Limitation
            self.Lchamber.change_length(-dL)
            self.Rchamber.change_length(dL)
            self.P_M = self.P_M_L_Limitation
        else:
            self.P_M = self.P_M - dL
            self.Lchamber.change_length(-dL)
            self.Rchamber.change_length(dL)
        return
    def move_motor_to_R(self, dL):
        if dL < 0:
            print('dL < 0')
            return
        if self.P_M + dL > self.P_M_R_Limitation:
            dL = self.P_M_R_Limitation - self.P_M
            self.Lchamber.change_length(dL)
            self.Rchamber.change_length(-dL)
            self.P_M = self.P_M_R_Limitation
        else:
            self.P_M = self.P_M + dL
            self.Lchamber.change_length(dL)
            self.Rchamber.change_length(-dL)
        return


class simulator:
    def __init__(self):
        return


if __name__ == '__main__':
    pump = hybrid_pump(L_L=0.1, L_R=0.1+0.1)
    print(pump.Lchamber_V_max, pump.Lchamber_V_min)
    print(pump.Rchamber_V_max, pump.Rchamber_V_min)
    pump.render()
    pump.move_motor_to_R(0.05)
    print(pump.Rchamber.P / P_0)
    pump.render()
    pump.move_motor_to_L(0.2)
    pump.render()
    pump.move_motor_to_R(0.1)
    pump.render()
    '''
    pump = hybrid_pump(0.1, 0.1)
    pump.move_motor_to_L(0.05)
    pump.open_L_valve()
    pump.close_L_valve()
    print('start:', 'self.pump.Lchamber.P:', pump.Lchamber.P, 'self.pump.Rchamber.P:', pump.Rchamber.P)

    # step 
    actions = np.array([[-0.11293268, -1.        ], [-0.73517716, -0.40774852], [ 1.        , -0.11729856], [ 0.3907593, -1.        ], [-0.68165046,  0.1595722 ], [0.4569448 , 0.09706157 ], [-0.87153196, -0.5983197 ]])
    for action in actions:
        valve = 0
        prev_P_M = pump.P_M
        goal_P_M = action[0] * 0.05
        move_distance = goal_P_M - prev_P_M
        if move_distance >= 0:
            pump.move_motor_to_R(move_distance)
        elif move_distance < 0:
            pump.move_motor_to_L(-move_distance)
        else:
            print("Action[0] error")
        if action[1] > 0.5 and action[1] <= 1:
            pump.open_R_valve()
            pump.close_R_valve()
            valve = 1
        elif action[1] > 0:
            pump.open_inner_valve()
            pump.close_inner_valve()
            valve = 2
        elif action[1] > -0.5:
            pump.open_L_valve()
            pump.close_L_valve()
            valve = 3
        elif action[1] >= -1:
            valve = 4
        else:
            print("Action[1] error")
            valve = 5
        print('action:', action, 'valve:', valve, 'self.pump.Lchamber.P:', pump.Lchamber.P, 'self.pump.Rchamber.P:', pump.Rchamber.P, 'self.pump.P_M:', pump.P_M)
    '''
    
    
    '''
    # (a)
    pump = hybrid_pump(0.1, 0.1+0.05)
    print(pump.P_M)
    # print(pump.Lchamber_V_max, pump.Lchamber_V_min)
    # print(pump.Rchamber_V_max, pump.Rchamber_V_min)
    print(pump.Lchamber.V, pump.Rchamber.V, pump.Lchamber.P, pump.Rchamber.P)
    pump.render()
    
    pump.move_motor_to_L(0.05)
    print(pump.P_M)
    print(pump.Lchamber.V, pump.Rchamber.V, pump.Lchamber.P, pump.Rchamber.P)
    pump.render()

    pump.move_motor_to_R(0.06)
    print('?',pump.P_M)
    print(pump.Lchamber.V, pump.Rchamber.V, pump.Lchamber.P, pump.Rchamber.P)
    pump.render()

    pump.move_motor_to_R(0.05)
    print(pump.P_M)
    print(pump.Lchamber.V, pump.Rchamber.V, pump.Lchamber.P, pump.Rchamber.P)
    pump.render()
    

    
    # Cycle
    # (b)
    pump.move_motor_to_L(0.05)
    pump.open_L_valve()
    pump.render()
    print(pump.Lchamber.V, pump.Rchamber.V, pump.Lchamber.P, pump.Rchamber.P)
    n_cycle = 3
    for i in range(n_cycle):
        # (c-d)
        pump.close_L_valve()
        pump.open_inner_valve()
        pump.render()
        # print(pump.Lchamber.V, pump.Rchamber.V, pump.Lchamber.P, pump.Rchamber.P)
        # (e-g)
        pump.close_inner_valve()
        pump.move_motor_to_R(0.05)
        pump.move_motor_to_R(0.05)
        pump.open_R_valve()
        pump.render()
        # print(pump.Lchamber.V, pump.Rchamber.V, pump.Lchamber.P, pump.Rchamber.P)
        # (h-j)
        pump.close_R_valve()
        pump.open_inner_valve()
        pump.render()
        # (k-b)
        pump.close_inner_valve()
        pump.move_motor_to_L(0.05)
        pump.move_motor_to_L(0.05)
        pump.open_L_valve()
        pump.render()
        print(pump.Lchamber.V, pump.Rchamber.V, pump.Lchamber.P, pump.Rchamber.P)

    output_P_min = pump.Rchamber.P
    print("output_P_min = ", output_P_min)  # 10635.948812228482
    '''
    '''
    # Inverse cycle
    # (b)
    pump.move_motor_to_L(0.05)
    pump.open_L_valve()
    pump.render()
    print(pump.Lchamber.V, pump.Rchamber.V, pump.Lchamber.P, pump.Rchamber.P)
    n_cycle = 3
    for i in range(n_cycle):
        # (k)
        pump.move_motor_to_R(0.05)
        pump.move_motor_to_R(0.05)
        pump.open_L_valve()
        pump.render()
        # (j-i)
        pump.close_L_valve()
        pump.open_inner_valve()
        pump.render()
        # (h-f)
        pump.close_inner_valve()
        pump.move_motor_to_L(0.05)
        pump.move_motor_to_L(0.05)
        pump.open_R_valve()
        pump.render()
        # (e-d)
        pump.close_R_valve()
        pump.open_inner_valve()
        pump.render()
        # (c-b)
        pump.close_inner_valve()
        pump.open_L_valve()
        print(pump.Lchamber.V, pump.Rchamber.V, pump.Lchamber.P, pump.Rchamber.P)
        
    output_P_max = pump.Rchamber.P
    print("output_P_max = ", output_P_max)  # 255661.6620987855
    '''
    

