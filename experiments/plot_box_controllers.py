import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np

# Plot a box plot of EPISODIC_REWARD vs test_loads
# [test_loads, EPISODIC_REWARD] = pickle.load( open( "./scripts/load_test_results.p", "rb" ) )




# # make data:
# np.random.seed(10)
# D = np.random.normal((3, 5, 4), (1.25, 1.00, 1.25), (100, 3))

cyan = '#8ECFC9'
orange = '#FFBE7A'
orange_1 = '#FA9F6F' # '#FA7F6F'
magenta = '#FA7F6F'
blue = '#82B0D2'
violet = '#BEB8DC'
beige = '#E7DAD2'
grey = '#999999'
grey_darker = '#555555'

# plot

style_name = 'seaborn-v0_8-whitegrid'
plt.style.use(style_name)

plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.size'] = 15

plt.rcParams['xtick.major.pad']='5'
plt.rcParams['ytick.major.pad']='5'
plt.rcParams['lines.linewidth']=2.5

COLOR = grey_darker
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR


filepaths_agent = []
filepaths_pid = []
filepaths_handcrafted = []
LOSS=[]
LOSS_HC=[]
LOSS_PID=[]

target_folders = ["deformable_small", "deformable_large", "non-deformable"]

for target_folder in target_folders:
    destination= f"./experiments/controllers/{target_folder}/overalL_results.svg"
    for ep in range(7):
        filepath_agent = f"./experiments/controllers/{target_folder}/saved_figures_real_agent/0/tracking_results_ep_{ep}.p"
        filepath_handcrafted = f"./experiments/controllers/{target_folder}/saved_figures_real_handcrafted/0/tracking_results_ep_{ep}.p"
        filepath_pid = f"./experiments/controllers/{target_folder}/saved_figures_real_pid/0/tracking_results_ep_{ep}.p"

        # [P_L, P_R, P_L_S, P_R_S, P_R_G, P_L_G, R, R_S, VL, VR] = pickle.load( open( "./scripts/save.p", "rb" ) )
        [P_L, P_R, _, _, P_R_G, P_L_G, R, R_S, VL, VR] = pickle.load( open( filepath_agent, "rb" ) )
        [P_L_HC, P_R_HC, _,_, P_R_G, P_L_G, R, R_S, _, _] = pickle.load( open( filepath_handcrafted, "rb" ) )
        [P_L_PID, P_R_PID, _,_, P_R_G, P_L_G, R, R_S, _, _] = pickle.load( open( filepath_pid, "rb" ) )

        P_L = np.array(P_L)
        # P_R = np.array(P_R)
        # P_L_HC = np.array(P_L_HC)
        P_L_HC = np.array([0]+P_L_HC)
        # P_R_HC = np.array(P_R_HC)
        P_L_PID = np.array([0]+P_L_PID)
        # P_R_PID = np.array(P_R_PID)
        # P_R_G = np.array(P_R_G)
        P_L_G = np.array(P_L_G[:100])
        VL=abs(VL)
        # VR=abs(VR)
        # print(P_L.shape, P_L_HC.shape, P_L_PID.shape, P_L_G.shape)
        loss = np.mean(abs(P_L_G-P_L))
        loss_HC = np.mean(abs(P_L_G-P_L_HC))
        loss_PID = np.mean(abs(P_L_G-P_L_PID))
        LOSS.append(loss)
        LOSS_HC.append(loss_HC)
        LOSS_PID.append(loss_PID)

    LOSS = np.array(LOSS)
    LOSS_HC = np.array(LOSS_HC)
    LOSS_PID = np.array(LOSS_PID)
    pickle.dump([LOSS, LOSS_HC,LOSS_PID], open(  f"./experiments/controllers/{target_folder}_losses.p", "wb" ) )

# print(LOSS.shape, LOSS_HC.shape, LOSS_PID.shape)
LOSS_MEAN, LOSS_STD = np.mean(LOSS), np.std(LOSS)
LOSS_HC_MEAN, LOSS_HC_STD = np.mean(LOSS_HC), np.std(LOSS_HC)
LOSS_PID_MEAN, LOSS_PID_STD = np.mean(LOSS_PID), np.std(LOSS_PID)

print(LOSS_MEAN, LOSS_STD, LOSS_HC_MEAN, LOSS_HC_STD, LOSS_PID_MEAN, LOSS_PID_STD)

# Loop 2    

fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(10,10))

# ax.axvspan(-0.5, 2.5, facecolor=cyan, alpha=0.2)
# ax.axvspan(2.5, 8.5, facecolor=magenta, alpha=0.2)

VP = ax.boxplot([LOSS, LOSS_HC, LOSS_PID], positions=[1,2,3], widths=0.5, patch_artist=True,
                showmeans=False, showfliers=False,
                medianprops={"color": grey_darker, "linewidth": 2},
                boxprops={"facecolor": violet, "edgecolor": grey_darker,
                        "linewidth": 2},
                whiskerprops={"color": grey_darker, "linewidth": 1.5},
                capprops={"color": grey_darker, "linewidth": 1.5})

ax.set_title("Comparison controllers with different loads")
# ax.set_xlabel("Loads")#, labelpad=2)
ax.set_ylabel("Loss (kPa)")

plt.xticks([1, 2, 3], ["RL", "Handcrafted+PID", "PID"])
# ax.set_ylabel("Loss (kPa)")



# ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
#         ylim=(0, 8), yticks=np.arange(1, 8))


# fig = plt.figure(figsize =(10, 7))
# ax = fig.add_axes(test_loads[:4])
# bp = ax.boxplot(LOAD_REWARD)

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



# fig.savefig('./saved_figures/overall_results.svg', format = 'svg')
fig.savefig(destination, format = 'svg')
# fig.show()

