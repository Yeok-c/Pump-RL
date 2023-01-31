import matplotlib.pyplot as plt
import pickle
import numpy as np

[P_L, P_R, P_R_G, P_L_G, R, VL, VR] = pickle.load( open( "./scripts/save.p", "rb" ) )
# RX = [i for i, val in enumerate(R) if val==5]
# RY = P[RX]
P_L = np.array(P_L)
P_L_G = np.array(P_L_G)
P_R = np.array(P_R)
P_R_G = np.array(P_R_G)
VL=abs(VL)
VR=abs(VR)

# R = np.array(R)
# RX = np.where(R >= 1)[0]
# RY = P[RX]

fig, ax = plt.subplots(1,1, figsize=(9, 4), dpi=100)
ax.plot(np.arange(0, P_L.size), P_L, '--b')
ax.plot(np.arange(1, P_L_G.size+1), P_L_G, '-b')
# ax.plot(RX, RY, 'or')
ax.plot(np.arange(0, P_R.size), P_R, '--r')
ax.plot(np.arange(1, P_R_G.size+1), P_R_G, '-r')

ax.set_title("Target and actuated pressures relative to atmosphere, \nV_L={:.03f}, \
    V_R={:.03f} (multiples of chamber volume)".format(VL, VR))
ax.set_xlabel("Timesteps")
ax.set_ylabel("kPa")

ax.legend(["P_L observed", "P_L goals", "P_R observed", "P_R goals"])
fig.savefig('./scripts/file.png')
# fig.show()

