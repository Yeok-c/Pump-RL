# The Simulator for the hybrid pump
import numpy
import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
from resources.graphics import Graphics

P_0 = 1.01*1e5  # Pa


class chamber:
    def __init__(self, L0, P0, load_volume=0, K_deform=0.05):
        # self.radius = 0.02  # Radius of chamber
        self.a = 0.03  # The short edge of chamber cross surface
        self.c = 0.03  # The long edge of chamber cross surface
        self.n = 6   # The number of element size
        self.L = L0  # Length of chamber
        self.P = P0  # Pressure of chamber
        self.deform_coeff = abs(np.random.normal(0, K_deform)) # Deformability always positive
        # self.deform_coeff > 0.05

        self.V = self.calculate_V()  # Volume of chamber
        self.P_prev = P0  # Pressure of chamber in previous step
        self.V_prev = self.V   # Volume of chamber in previous step

        self.load_V = load_volume # load of chamber does not have its own load
        self.load_P = P_0
        self.load_valve = 0        
        return
    
    def calculate_V(self, L=None):
        if L is None:
            L = self.L
        # Calculate the volume of chamber (V) by the length (L)
        compensate = -2.5E-06
        # a = self.a*100
        # c = self.c*100
        # L = L*100
        
        a = self.a
        c = self.c
        L = L

        h = L / self.n / 2
        V_seg = 2 * a * a * h + math.sqrt(2) * a * c * h
        # V_seg = 2 * self.a * self.a * h + math.sqrt(2) * self.a * self.c * h
        V_seg = V_seg + (math.sqrt(2) * a + 2.0/3 * c) * math.sqrt(c*c - 2*h*h) * h
        # V = self.n * V_seg + compensate
        V = self.n * V_seg 
        V += compensate
        # V = V/1000000 + compensate

        V += (self.P-P_0)/P_0*V*self.deform_coeff
        # if P>P_0 (postitive pressure), term is postiive (more V than expected)
        # if P<P_0 (negative pressure), term is negative (less V than expected)
        
        if V <= 0:
            raise AssertionError (
            # print(
                "V is negative. V:{}, P:{}, Kd:{}, V before deform:{}".format(
                    V, self.P, self.deform_coeff, V-(self.P-P_0)/P_0*V*self.deform_coeff))


        return V
        
    def calculate_P(self):
        # Calculate the pressure of chamber (P) by the volume (V)
        # return self.P_prev * (self.V_prev + self.load_V / (self.V + self.load_V)
        P = self.P_prev * self.V_prev / self.V
        return P

    def change_length(self, dL):
        self.V_prev = self.V
        self.P_prev = self.P
        self.L = self.L + dL
        self.V = self.calculate_V()  # update V
        self.P = self.calculate_P()  # update P

        if self.load_valve == 1: # If load valve is open 
            #Equalize pressures between chamber and connected load
            self.equalize_pressures_with_load() 
        return

    def equalize_pressures_with_load(self):
        P_new = (self.P * self.V + self.load_P * self.load_V) / (
            self.V + self.load_V)
        self.P = P_new
        self.load_P = P_new
        return

#      | |              | | 
#  IO3 > <              > < IO4
#      | |              | | 
# \/__|——————|__\/__|——————|__\/
# /\  |——————|  /\  |——————|  /\
# IO0  VL, PL   IO1  VR, PR   IO2

class hybrid_pump:
    def __init__(self, L_L, L_R, load_chamber_ratio_L, load_chamber_ratio_R, K_deform):
        self.P_M = 0  # motor
        self.P_M_L_Limitation = -0.015
        self.P_M_R_Limitation = +0.015
        
        self.testchamber = chamber(L_L, P_0)
        V_0 = self.testchamber.V 
        self.V_L = 0
        self.V_R = 0

        self.Lchamber = chamber(L_L, P_0, load_volume=load_chamber_ratio_L*V_0, K_deform=K_deform)
        self.Rchamber = chamber(L_R, P_0, load_volume=load_chamber_ratio_R*V_0, K_deform=K_deform)
        self.valve = [0,0,0,0,0]  # 0: close, 1: open  # L, R, inner, load L, load R

        self.Lchamber_V_max = self.Lchamber.calculate_V(L_L + abs(self.P_M_R_Limitation))
        self.Rchamber_V_max = self.Rchamber.calculate_V(L_R + abs(self.P_M_L_Limitation))
        self.Lchamber_V_min = self.Lchamber.calculate_V(L_L - abs(self.P_M_L_Limitation))
        self.Rchamber_V_min = self.Rchamber.calculate_V(L_R - abs(self.P_M_R_Limitation))
        
    def render(self, time=0, title='', filename=None, render_chamber_pressures=False):
        # MM_2_PX = 1000
        # self.img = np.zeros((200,400,3),dtype='uint8')
        # cv2.rectangle(self.img,(100,100),(100+int(0.2*MM_2_PX),100-10),(0,255,0),3)  # pump
        # cv2.circle(self.img,(int(200+self.P_M*MM_2_PX),100-5),8,(255,0,0),-1)  # motor
        
        # if self.valve[0] == 0:
        #     cv2.circle(self.img,(90,100-5),3,(0,0,255),-1)  # IO0
        # if self.valve[1] == 0:
        #     cv2.circle(self.img,(310,100-5),3,(0,0,255),-1)   # IO1
        # if self.valve[2] == 0:
        #     cv2.circle(self.img,(200,100+10),3,(0,0,255),-1)  # IO2
        # if self.valve[3] == 0:
        #     cv2.circle(self.img,(130,80),3,(0,0,255),-1)  # IO3
        # if self.valve[4] == 0:
        #     cv2.circle(self.img,(270,80),3,(0,0,255),-1)  # IO4

        # cv2.putText(self.img,'Chamber_L P: {p:.2f}'.format(p=self.Lchamber.P/P_0),(30,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
        # cv2.putText(self.img,'Chamber_R P: {p:.2f}'.format(p=self.Rchamber.P/P_0),(230,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
        # cv2.putText(self.img,'Load_L P: P: {p:.2f}'.format(p=self.Lchamber.load_P/P_0),(60,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
        # cv2.putText(self.img,'Load_R P: {p:.2f}'.format(p=self.Rchamber.load_P/P_0),(230,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
        # cv2.putText(self.img, title,(50,180), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
        # cv2.imshow('a',self.img)

        self.graphics = Graphics() 
        self.graphics.render_valve_and_motor(self.valve, self.P_M)


        if render_chamber_pressures == True:
            self.graphics.add_text_to_image('   Left chamber \npressure: {: 06.2F} kPa'.format((self.Lchamber.P-P_0)/1000), (220,120))
            self.graphics.add_text_to_image('   Right chamber \npressure: {: 06.2F} kPa'.format((self.Rchamber.P-P_0)/1000), (510,120))
        else:
            self.graphics.add_text_to_image('     \nLeft chamber', (220,120))
            self.graphics.add_text_to_image(' \nRight chamber', (510,120))

        self.graphics.add_text_to_image('     Left load \npressure: {: 06.2F} kPa'.format((self.Lchamber.load_P-P_0)/1000), (15,300))
        self.graphics.add_text_to_image('     Right load \npressure: {: 06.2F} kPa'.format((self.Rchamber.load_P-P_0)/1000), (665,300))

        self.graphics.add_text_to_image(title, (100,100), font_scale=0.7)
       
        if filename==None:
            pass
        else:
            cv2.imwrite(filename, self.graphics.display_image)

        if time != 0:
            cv2.imshow('a',self.graphics.display_image)
            cv2.waitKey(time)

        return self.graphics.display_image

    def equalize_pressures(self, chamber1, chamber2):
        P_new = (chamber1.P * chamber1.V + chamber2.P * chamber2.V) / (chamber1.V + chamber2.V)
        return P_new, P_new

    def open_L_valve(self):
        self.valve[1] = 1
        self.Lchamber.P = P_0
        return
    def close_L_valve(self):
        self.valve[1] = 0
        return

    def open_R_valve(self):
        self.valve[2] = 1
        self.Rchamber.P = P_0
        return
    def close_R_valve(self):
        self.valve[2] = 0
        return

    def open_inner_valve(self):
        self.valve[4] = 1
        self.Lchamber.P, self.Rchamber.P = self.equalize_pressures(self.Lchamber, self.Rchamber)
        return
    def close_inner_valve(self):
        self.valve[4] = 0
        return

    def open_L_load_valve(self):
        self.valve[0] = 1
        self.Lchamber.load_valve = 1
        self.Lchamber.equalize_pressures_with_load()
        return
    def close_L_load_valve(self):
        self.valve[0] = 0
        self.Lchamber.load_valve = 0
        return
    
    def open_R_load_valve(self):
        self.valve[3] = 1        
        self.Rchamber.load_valve = 1
        self.Rchamber.equalize_pressures_with_load()
        return
    def close_R_load_valve(self):
        self.Rchamber.load_valve = 0
        self.valve[3] = 0
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
    pump = hybrid_pump(
        L_L=0.1, L_R=0.1, 
        load_chamber_ratio_L=1.5, 
        load_chamber_ratio_R=0.8,
        K_deform=0
    )

    print(pump.Lchamber_V_max, pump.Lchamber_V_min)
    print(pump.Rchamber_V_max, pump.Rchamber_V_min)
    pump.render(title='step 1', time=1)

    pump.move_motor_to_R(0.015)
    print('start:', 'self.pump.Lchamber.P:', pump.Lchamber.P, 'self.pump.Rchamber.P:', pump.Rchamber.P)
    pump.render(time=1000)

    pump.move_motor_to_L(0.015)
    print('start:', 'self.pump.Lchamber.P:', pump.Lchamber.P, 'self.pump.Rchamber.P:', pump.Rchamber.P)
    pump.render(time=1000)

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
    

