import matplotlib.pyplot as plt
import pickle
import numpy as np

[load, rew_mean, rew_std] = pickle.load( open( "./scripts/load_test_results.p", "rb" ) )
load = np.array(load)
rew_mean = np.array(rew_mean)
rew_std = np.array(rew_std)

#  0   1   2  3  4  5  6  7  8  9
# [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8]

# (0,1), (0,2), (0,3) etc
# average_list = np.array([
#     np.arange(0,3),  # (0, 1)
#     np.arange(0,4),  # (0, 2)
#     np.arange(0,5),  # (0, 3)
#     np.arange(0,6),  # (0, 4)
#     np.arange(0,10),  # (0, 8)
# ])

# (0,1), (1,2), (2,3) etc
average_list = np.array([
    np.arange(0,2),  # (0, 1)
    np.arange(2,3),  # (1, 2)
    np.arange(3,4),  # (2, 3)
    np.arange(4,5),  # (3, 4)
    np.arange(5,10),  # (4, 8)
])

for item in average_list:
    print(np.mean(rew_mean[item]))

print(rew_mean)

# fig, ax = plt.subplots(1,1)

# ax.plot(load, rew_mean)
# ax.set_title("Reward mean with various loads")
# ax.set_xlabel("Load volume (as a multiple of chamber volume)")
# ax.set_ylabel("Reward mean")

# fig, ax = plt.subplots(2,1)
# ax=ax.flatten()
# ax[0].plot(load, rew_mean)
# ax[0].set_title("Reward mean and various loads")
# ax[0].set_xlabel("Load")
# ax[0].set_ylabel("Reward mean")

# ax[1].plot(load, rew_std)
# ax[1].set_title("Reward std and various loads")
# ax[1].set_xlabel("Load")
# ax[1].set_ylabel("Reward std")



# fig.savefig('./scripts/file.png')
# fig.show()

