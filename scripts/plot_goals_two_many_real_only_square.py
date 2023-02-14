import matplotlib.pyplot as plt
import pickle
import numpy as np
import glob


'''
Script to zip all these files
zip -r ./saved_figures.zip ./saved_figures** ./scripts/plot_goals_two_many_real_only_square.py -x ./saved_figures**/**\step_** 

'''

cyan = '#8ECFC9'
orange = '#FFBE7A'
orange_1 = '#FA9F6F' # '#FA7F6F'
magenta = '#FA7F6F'
blue = '#82B0D2'
violet = '#BEB8DC'
beige = '#E7DAD2'
grey = '#999999'
grey_darker = '#444444'

# for style_name in plt.style.available:
# style_name = 'seaborn-v0_8-whitegrid'
# plt.style.use(style_name)
# for style_name in plt.style.available:
style_name = 'seaborn-v0_8-whitegrid'
plt.style.use(style_name)

plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.sans-serif'] = "Times New Roman"
plt.rcParams['font.size'] = 25

plt.rcParams['xtick.major.pad']='5'
plt.rcParams['ytick.major.pad']='5'
plt.rcParams['lines.linewidth']=2.5


COLOR = grey_darker
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR


filepaths = glob.glob("./saved_figures**/*/" + "*.p")
print(filepaths)

for file in filepaths:
    # [P_L, P_R, P_L_S, P_R_S, P_R_G, P_L_G, R, R_S, VL, VR] = pickle.load( open( "./scripts/save.p", "rb" ) )
    [P_L, P_R, P_L_S, P_R_S, P_R_G, P_L_G, R, R_S, VL, VR] = pickle.load( open( file, "rb" ) )
    # RX = [i for i, val in enumerate(R) if val==5]
    # RY = P[RX]
    P_L = np.array(P_L)
    P_L_S = np.array(P_L_S)
    P_L_G = np.array(P_L_G)

    P_R = np.array(P_R)
    P_R_S = np.array(P_R_S)
    P_R_G = np.array(P_R_G)
    VL=abs(VL)
    VR=abs(VR)

    # R = np.array(R)
    # RX = np.where(R >= 1)[0]
    # RY = P[RX]

    fig, ax = plt.subplots(1,1)
    fig.subplots_adjust(bottom=0.22, left=0.22)
    # fig, ax = plt.subplots(1,1, figsize=(9, 4), dpi=100)

    ax.grid(axis='y', linewidth=1.5, alpha=0.4)
    ax.grid(axis='x',alpha=0)

    ax.set_xlim([1, 100])
    # ax.set_xlabel("Timesteps", labelpad=12) # , weight='bold')
    # ax.set_ylabel("Pressure (kPa)", labelpad=12) #, weight='bold')


    ax.plot(np.arange(1, P_L.size+1), P_L, linestyle='dashed', color=blue) # '--b')
    # ax.plot(np.arange(1, P_L_S.size+1), P_L_S, linestyle='dashed', color=cyan) # '--b')
    ax.plot(np.arange(2, P_L_G.size+2), P_L_G, linestyle='solid', color=blue) # '-b')
    # ax.plot(RX, RY, 'or')
    ax.plot(np.arange(1, P_R.size+1), P_R, linestyle='dashed', color=orange_1) # '--r')
    # ax.plot(np.arange(1, P_R_S.size+1), P_R_S, linestyle='dashed', color=orange) # '--r')
    ax.plot(np.arange(2, P_R_G.size+2), P_R_G, linestyle='solid', color=orange_1) #'-r')



    ax.grid(axis='y', linewidth=1.5, alpha=0.4)
    ax.grid(axis='x',alpha=0)

    ax.set_xlabel("Timesteps", labelpad=12) # , weight='bold')
    ax.set_ylabel("Pressure (kPa)", labelpad=12) #, weight='bold')
    # fig.supxlabel("Timesteps")
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)

    ax.set_xlim([0, 100])

    # ax.set_title("Target and actuated pressures relative to atmosphere, \nV_L={:.01f}, \
    #     V_R={:.01f} (multiples of chamber volume)".format(VL, VR))
    # ax.set_xlabel("Timesteps", fontsize=16)
    # ax.set_ylabel("kPa", fontsize=16)

    # ax.legend(["P_L observed real", "P_L observed sim", "P_L goals", "P_R observed real", "P_R observed sim", "P_R goals"])
    fig.savefig(file + '_real_only.svg', format = 'svg', dpi=300)
    plt.close()
    # plt.savefig("./scripts/file.svg", format = 'svg', dpi=300)
    
    # fig.show()

