import matplotlib.pyplot as plt
import pickle
import numpy as np

[P, P_G, R] = pickle.load( open( "save.p", "rb" ) )
# RX = [i for i, val in enumerate(R) if val==5]
# RY = P[RX]
P = np.array(P)
P_G = np.array(P_G)
R = np.array(R)
RX = np.where(R >= 1)[0]
RY = P[RX]
fig, ax = plt.subplots(1,1)
ax.plot(np.arange(0, P.size), P)
ax.plot(np.arange(1, P_G.size+1), P_G)
ax.plot(RX, RY, 'or')
ax.legend(["pressure observed", "pressure goals"])
fig.savefig('file.png')
# fig.show()

