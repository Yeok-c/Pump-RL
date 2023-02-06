import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np



cyan = '#8ECFC9'
orange = '#FFBE7A'
orange_1 = '#FA9F6F' # '#FA7F6F'
magenta = '#FA7F6F'
blue = '#82B0D2'
violet = '#BEB8DC'
beige = '#E7DAD2'
grey = '#999999'
grey_darker = '#444444'


def load_experiment_results(filepath = "./scripts/chamber_experiments_real.p"):
    [P_M_, LCHAMBER_P_, RCHAMBER_P_] = pickle.load( open( filepath, "rb" ) )
    P_M_=np.array(P_M_) 
    LCHAMBER_P_=np.array(LCHAMBER_P_) 
    RCHAMBER_P_=np.array(RCHAMBER_P_)
    return P_M_, LCHAMBER_P_, RCHAMBER_P_

fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(10, 10), dpi=300)
ax=ax.flatten()

P_M_, LCHAMBER_P_, RCHAMBER_P_ = load_experiment_results(filepath="./scripts/chamber_experiments_real_range_010.p")
# ax[0].plot(P_M_, linestyle='dashed', color=violet, alpha=0.5)
ax[0].plot(LCHAMBER_P_, linestyle='dashed', color=violet, alpha=0.5)
ax[1].plot(RCHAMBER_P_, linestyle='dashed', color=violet, alpha=0.5)

P_M_, LCHAMBER_P_, RCHAMBER_P_ = load_experiment_results(filepath="./scripts/chamber_experiments_sim_range_009.p")
# ax[0].plot(P_M_, linestyle='solid', color=cyan, alpha=0.5)
ax[0].plot(LCHAMBER_P_, linestyle='solid', color=cyan, alpha=0.5)
ax[1].plot(RCHAMBER_P_, linestyle='solid', color=cyan, alpha=0.5)

P_M_, LCHAMBER_P_, RCHAMBER_P_ = load_experiment_results(filepath="./scripts/chamber_experiments_real_range_009.p")
# ax[0].plot(P_M_, linestyle='dashed', color=cyan, alpha=0.5)
ax[0].plot(LCHAMBER_P_, linestyle='dashed', color=cyan, alpha=0.5)
ax[1].plot(RCHAMBER_P_, linestyle='dashed', color=cyan, alpha=0.5)

P_M_, LCHAMBER_P_, RCHAMBER_P_ = load_experiment_results(filepath="./scripts/chamber_experiments_sim_range_008.p")
# ax[0].plot(P_M_, linestyle='solid', color=grey)
ax[0].plot(LCHAMBER_P_, linestyle='solid', color=grey)
ax[1].plot(RCHAMBER_P_, linestyle='solid', color=grey)

P_M_, LCHAMBER_P_, RCHAMBER_P_ = load_experiment_results(filepath="./scripts/chamber_experiments_real_009_motor_r_038.p")
# ax[0].plot(P_M_, linestyle='dashed', color=magenta, alpha=0.5)
ax[0].plot(LCHAMBER_P_, linestyle='dashed', color=magenta, alpha=0.5)
ax[1].plot(RCHAMBER_P_, linestyle='dashed', color=magenta, alpha=0.5)

P_M_, LCHAMBER_P_, RCHAMBER_P_ = load_experiment_results(filepath="./scripts/chamber_experiments_real_009_motor_r_035.p")
# ax[0].plot(P_M_, linestyle='dashed', color=grey)
ax[0].plot(LCHAMBER_P_, linestyle='dashed', color=grey)
ax[1].plot(RCHAMBER_P_, linestyle='dashed', color=grey)


ax[0].legend([
    "Real, range 0.010", 
    "Sim, range 0.009", 
    "Real, range 0.009", 
    "Sim, range 0.008", 
    "Real, range 0.009, motor_r 0.038",
    "Real, range 0.009, motor_r 0.035",
])

ax[1].legend([
    "Real, range 0.010", 
    "Sim, range 0.009", 
    "Real, range 0.009", 
    "Sim, range 0.008",         
    "Real, range 0.009, motor_r 0.038",
    "Real, range 0.009, motor_r 0.035",

])

# ax[2].legend([
#     "Real, range 0.010", 
#     "Sim, range 0.009", 
#     "Real, range 0.009", 
#     "Sim, range 0.008", 
#     "Real, range 0.009, motor_r 0.038",
#     "Real, range 0.009, motor_r 0.035",
# ])

# ax[0].set_title("Pump Motor Position")
ax[0].set_title("Left Chamber Pressure")
ax[1].set_title("Right Chamber Pressure")


fig.savefig('./scripts/changed_params_pressures.svg', format = 'svg')
fig.savefig('./scripts/changed_params_pressures.png', format = 'png')
plt.show()