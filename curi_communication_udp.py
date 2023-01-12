import sys
import select
import socket
class curi_communication_udp:
    name = 'udp'
    def __init__(self, selfIP, selfPort, targetIP, targetPort):
        self.self_IP = selfIP
        self.self_Port = selfPort
        self.target_Address = (targetIP, targetPort)
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
        return
    
    def open(self):
        # self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind((self.self_IP, self.self_Port))
        print('open socket')

    def close(self):
        print('close socket')
        self.s.close()

    def send(self, massage):
        self.s.sendto(massage.encode("utf-8"), self.target_Address)
        
    def recieve(self, dt = 0.001): # waiting time
        readable = select.select([self.s], [], [], dt)[0]
        buf = ""
        if readable:
            for a in readable:
                buf = a.recvfrom(256)[0].decode("utf-8")
                # print("Received:", buf)
                # buf = str(a.recvfrom(256)[0])
        return buf

    def set_start(self):
        self.send("start")

    def set_stop(self):
        self.send("stop")

import time
if __name__ == '__main__':
    CS = curi_communication_udp("127.0.0.1", 13331, "127.0.0.1", 13332)
    CS.open()
    for i in range(100):
        rc = CS.recieve()
        if rc != "":
            print('recieve:', CS.recieve())
        CS.send(str(i))
        # time.sleep(0.05)
    CS.close()
