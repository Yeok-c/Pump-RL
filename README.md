# Pump-RL
# UDP 
## UDP
Python udp code: **pump_real_load_addvalve_dra.py**
### 1. Get pressure readings
+ Python code uses **getPressure** string to initialize the pump
```
udp = curi_communication_udp("127.0.0.1", 13331, "127.0.0.1", 13332)
udp.send("getPressure")
rc = udp.recieve()
if rc != "":
    print('recieve:', rc)
P1 = float(rc.split(',')[0])
P2 = float(rc.split(',')[1])
P3 = float(rc.split(',')[2])
P4 = float(rc.split(',')[3])
```
Expected effects:
```
C code will let Python code know four pressure sensors readings in the format of:
rc = "p1,p2,p3,p4"
```

### 2. Get piston position
+ Python code uses **getPosition** string to initialize the pump
```
udp = curi_communication_udp("127.0.0.1", 13331, "127.0.0.1", 13332)
udp.send("getPosition")
rc = udp.recieve()
if rc != "":
    print('recieve:', rc)
P_M = float(rc)
```
Expected effects:
```
C code will let Python code know piston position in mm:
e.g., "0.5" means current piston is at +0.5 mm
```

### 3. Set piston position
+ Python code uses **set_position** string to move the piston
```
udp = curi_communication_udp("127.0.0.1", 13331, "127.0.0.1", 13332)
udp.send("setPosition,"+str(position))
# Wait for moving done
moving_done = False
while moving_done == False:
    rc = self.udp.recieve()
    if rc == "moving done":
        moving_done = True
return moving_done
```
Expected effects:
```
C code will control the piston position instructed by Python code:
udp.send("setPosition,"+str(position))
When motor finishes, C code will return a string "moving done" to Python code.
```

### 4. Get valve states
+ Python code uses **get_valves** string to get valve states
```
udp = curi_communication_udp("127.0.0.1", 13331, "127.0.0.1", 13332)
udp.send("getValve")
rc = self.udp.recieve()
if rc != "":
    print('recieve:', rc)
self.valve[0] = int(rc.split(',')[0])
self.valve[1] = int(rc.split(',')[1])
self.valve[2] = int(rc.split(',')[2])
self.valve[3] = int(rc.split(',')[3])
self.valve[4] = int(rc.split(',')[4])
return self.valve
```
Expected effects:
```
C code will return IO states in the format of:
[IO0,IO1,IO2,IO3,IO4]
```

### 5. Set valve states
+ Python code uses **setValve** string to get valve states
In the following example, valve is a list of 5 binary values.
```
udp = curi_communication_udp("127.0.0.1", 13331, "127.0.0.1", 13332)
udp.send("setValve,"+str(valve[0])+","+str(valve[1])+","+str(valve[2])+","+str(valve[3])+","+str(valve[4]))
# Wait for moving done
IO_done = False
while IO_done == False:
    rc = self.udp.recieve()
    if rc == "IO done":
        IO_done = True
```
Expected effects:
```
C code will control IO states.
Once done, C code will return a string "IO done" to Python code.
```

