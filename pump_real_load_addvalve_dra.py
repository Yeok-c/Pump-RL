# Communication to the real pump
import numpy
import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
from curi_communication_udp import curi_communication_udp
import time


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
        
#     def calculate_P(self):
#         # Calculate the pressure of chamber (P) by the volume (V)
#         # return self.P_prev * (self.V_prev + self.load_V / (self.V + self.load_V)
#         P = self.P_prev * self.V_prev / self.V
#         return P

#     def change_length(self, dL):
#         self.V_prev = self.V
#         self.P_prev = self.P
#         self.L = self.L + dL
#         self.V = self.calculate_V()  # update V
#         self.P = self.calculate_P()  # update P

#         if self.load_valve == 1: # If load valve is open 
#             #Equalize pressures between chamber and connected load
#             self.equalize_pressures_with_load() 
#         return

#     def equalize_pressures_with_load(self):
#         P_new = (self.P * self.V + self.load_P * self.load_V) / (
#             self.V + self.load_V)
#         self.P = P_new
#         self.load_P = P_new
#         return

#      | |              | | 
#  IO3 > <              > < IO4
#      | |              | | 
# \/__|——————|__\/__|——————|__\/
# /\  |——————|  /\  |——————|  /\
# IO0  VL, PL   IO2  VR, PR   IO1

class real_pump:
    def __init__(self):
        # real
        self.P_M = 0  # motor
        self.P_M_L_Limitation = -0.05
        self.P_M_R_Limitation = +0.05
        self.valve = [0,0,0,0,0]  # 0: close, 1: open  [L, R, inner, load L, load R]
        self.pressure = [0,0,0,0]  # [Lc, Rc, load L, load R]

        # real pump init
        self.udp = curi_communication_udp("127.0.0.1", 13331, "127.0.0.1", 13332)
        self.udp.open()
        # self.set_valves([1,1,1,1,1])
        # # self.set_position(0)
        # self.set_valves([0,0,0,0,0])
        # self.get_position()
        # self.get_pressure()
        # self.get_valves()

        # Estimate init chamber volume
        self.testchamber = chamber(0.1, P_0)
        V_0 = self.testchamber.V 
        self.V_L = 0
        self.V_R = 0
        self.Lchamber_V = V_0   # real pump TODO
        self.Rchamber_V = V_0 

        # self.Lchamber = chamber(L_L, P_0, load_volume=load_chamber_ratio_L*V_0, K_deform=K_deform)
        # self.Rchamber = chamber(L_R, P_0, load_volume=load_chamber_ratio_R*V_0, K_deform=K_deform)
        # self.Lchamber_V_max = self.Lchamber.calculate_V(L_L + abs(self.P_M_R_Limitation))
        # self.Rchamber_V_max = self.Rchamber.calculate_V(L_R + abs(self.P_M_L_Limitation))
        # self.Lchamber_V_min = self.Lchamber.calculate_V(L_L - abs(self.P_M_L_Limitation))
        # self.Rchamber_V_min = self.Rchamber.calculate_V(L_R - abs(self.P_M_R_Limitation))
        
        
    # def render(self, time=0, title='', filename=None):
    #     MM_2_PX = 1000
    #     self.img = np.zeros((200,400,3),dtype='uint8')
    #     cv2.rectangle(self.img,(100,100),(100+int(0.2*MM_2_PX),100-10),(0,255,0),3)  # pump
    #     cv2.circle(self.img,(int(200+self.P_M*MM_2_PX),100-5),8,(255,0,0),-1)  # motor
        
    #     if self.valve[0] == 0:
    #         cv2.circle(self.img,(90,100-5),3,(0,0,255),-1)  # IO0
    #     if self.valve[1] == 0:
    #         cv2.circle(self.img,(310,100-5),3,(0,0,255),-1)   # IO1
    #     if self.valve[2] == 0:
    #         cv2.circle(self.img,(200,100+10),3,(0,0,255),-1)  # IO2
    #     if self.valve[3] == 0:
    #         cv2.circle(self.img,(130,80),3,(0,0,255),-1)  # IO3
    #     if self.valve[4] == 0:
    #         cv2.circle(self.img,(270,80),3,(0,0,255),-1)  # IO4

    #     cv2.putText(self.img,'Chamber_L P: {p:.2f}'.format(p=self.Lchamber.P/P_0),(30,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
    #     cv2.putText(self.img,'Chamber_R P: {p:.2f}'.format(p=self.Rchamber.P/P_0),(230,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
    #     cv2.putText(self.img,'Load_L P: P: {p:.2f}'.format(p=self.Lchamber.load_P/P_0),(60,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
    #     cv2.putText(self.img,'Load_R P: {p:.2f}'.format(p=self.Rchamber.load_P/P_0),(230,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
    #     cv2.putText(self.img, title,(50,180), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
    #     cv2.imshow('a',self.img)
    #     if filename==None:
    #         pass
    #     else:
    #         cv2.imwrite(filename, self.img)
    #     cv2.waitKey(time)


    def get_pressure(self):
        msg = "getPr"
        self.udp.send(msg)
        print("Sent: ", msg)

        # Sample received string: recPr:0.000000,0.000000,0.000000,0.000000"
        rc = self.udp.recieve()
        rc = rc.split(':')

        if rc[0] == "recPr":
            rc = rc[1]
            P1 = float(rc.split(',')[0])
            P2 = float(rc.split(',')[1])
            P3 = float(rc.split(',')[2])
            P4 = float(rc.split(',')[3])
            self.pressure = [P1, P2, P3, P4]
            print('Received pressures:', self.pressure)
        return self.pressure

    def get_position(self):
        msg = "getPo"
        self.udp.send(msg)
        print("Sent: ", msg)        

        # Sample received string: recPo:0.000020 
        rc = self.udp.recieve()
        rc = rc.split(':')
        
        if rc[0] == "recPo":
            rc = rc[1]
            print('Received piston position:', rc)
            self.P_M = float(rc)

        return self.P_M

    def set_position(self, position):
        msg = "setPo:"+str(position)
        self.udp.send(msg)
        print("Sent: ", msg)
        # Wait for moving done
        moving_done = False
        while moving_done == False:
            rc = self.udp.recieve()
            if rc == "moving done": ## !!!!!!!!!!!
                moving_done = True
        self.get_position()
        return moving_done

    def get_valves(self):
        msg = "getVa"
        self.udp.send(msg)
        print("Sent: ", msg)

        rc = self.udp.recieve()
        try:
            rc = rc.split('\'')[1]
        except:
            pass
        if rc != "" and rc != "moving done":
            print('recieve:', rc)

            self.valve[0] = int(rc.split(',')[0])
            self.valve[1] = int(rc.split(',')[1])
            self.valve[2] = int(rc.split(',')[2])
            self.valve[3] = int(rc.split(',')[3])
            self.valve[4] = int(rc.split(',')[4])
        return self.valve

    def set_valves(self, valve):
        msg = "setVa,"+str(valve[0])+","+str(valve[1])+","+str(valve[2])+","+str(valve[3])+","+str(valve[4])
        self.udp.send(msg)
        print("Sent: ", msg)

        # Wait for moving done
        IO_done = False
        while IO_done == False:
            time.sleep(0.1)
            rc = self.udp.recieve()
            rc = rc[2:-1]
            print(rc)
            if rc == 'IO done':
                IO_done = True
        self.get_valves()
        return

    # def equalize_pressures(self, chamber1, chamber2):
    #     P_new = (chamber1.P * chamber1.V + chamber2.P * chamber2.V) / (chamber1.V + chamber2.V)
    #     return P_new, P_new

    def open_L_valve(self):
        self.get_valves()
        self.valve[0] = 1
        self.set_valves(self.valve)
        # self.Lchamber.P = P_0
        return

    def close_L_valve(self):
        self.get_valves()
        self.valve[0] = 0
        self.set_valves(self.valve)
        return

    def open_R_valve(self):
        self.get_valves()
        self.valve[1] = 1
        # self.Rchamber.P = P_0
        self.set_valves(self.valve)
        return

    def close_R_valve(self):
        self.get_valves()
        self.valve[1] = 0
        self.set_valves(self.valve)
        return

    def open_inner_valve(self):
        self.get_valves()
        self.valve[2] = 1
        # self.Lchamber.P, self.Rchamber.P = self.equalize_pressures(self.Lchamber, self.Rchamber)
        self.set_valves(self.valve)
        return

    def close_inner_valve(self):
        self.get_valves()
        self.valve[2] = 0
        self.set_valves(self.valve)
        return

    def open_L_load_valve(self):
        self.get_valves()
        self.valve[3] = 1
        # self.Lchamber.load_valve = 1
        # self.Lchamber.equalize_pressures_with_load()
        self.set_valves(self.valve)
        return

    def close_L_load_valve(self):
        self.get_valves()
        self.valve[3] = 0
        # self.Lchamber.load_valve = 0
        self.set_valves(self.valve)
        return
    
    def open_R_load_valve(self):
        self.get_valves()
        self.valve[4] = 1        
        # self.Rchamber.load_valve = 1
        # self.Rchamber.equalize_pressures_with_load()
        self.set_valves(self.valve)
        return

    def close_R_load_valve(self):
        self.get_valves()
        # self.Rchamber.load_valve = 0
        self.valve[4] = 0
        self.set_valves(self.valve)
        return

    def move_motor_to_L(self, dL):
        self.get_position()
        if dL < 0:
            print('dL < 0')
            return
        if self.P_M - dL < self.P_M_L_Limitation:
            # Move to the far left
            dL = self.P_M - self.P_M_L_Limitation
            # self.Lchamber.change_length(-dL)
            # self.Rchamber.change_length(dL)
            self.P_M = self.P_M_L_Limitation
            self.set_position(self.P_M)
        else:
            # Move to the left
            self.P_M = self.P_M - dL
            # self.Lchamber.change_length(-dL)
            # self.Rchamber.change_length(dL)
            self.set_position(self.P_M)
        return

    def move_motor_to_R(self, dL):
        if dL < 0:
            print('dL < 0')
            return
        if self.P_M + dL > self.P_M_R_Limitation:
            # Move to the far right
            dL = self.P_M_R_Limitation - self.P_M
            # self.Lchamber.change_length(dL) 
            # self.Rchamber.change_length(-dL)
            self.P_M = self.P_M_R_Limitation
            print("setting position to: ", self.P_M)
            self.set_position(self.P_M)
        else:
            # Move to the right
            self.P_M = self.P_M + dL
            # self.Lchamber.change_length(dL)
            # self.Rchamber.change_length(-dL)
            print("setting position to: ", self.P_M)
            self.set_position(self.P_M)
        return


if __name__ == '__main__':
    # Test code for udp
    pump = real_pump()
    while(1):
        pump.get_pressure()
        time.sleep(1)
        pump.get_position()
        time.sleep(1)
    # pump.get_valves()
    # print(pump.valve)
    # pump.open_L_valve()
    # print(pump.valve)
    # pump.close_L_valve()
    # print(pump.valve)
    # pump.open_R_valve()
    # print(pump.valve)
    # pump.close_R_valve()
    # print(pump.valve)
    # pump.open_inner_valve()
    # print(pump.valve)
    # pump.close_inner_valve()
    # print(pump.valve)
    # pump.open_L_load_valve()
    # print(pump.valve)
    # pump.close_L_load_valve()
    # print(pump.valve)
    # pump.open_R_load_valve()
    # print(pump.valve)
    # pump.close_R_load_valve()
    # print(pump.valve)
    # pump.get_position()
    # print(pump.P_M)
    # pump.move_motor_to_L(0.014)
    # pump.get_position()
    # print(pump.P_M)
    # pump.move_motor_to_R(0.014)
    # pump.get_position()
    # print(pump.P_M)

