import matplotlib.pyplot as plt
import pickle
import numpy as np
import glob
import os

'''
zip -r ./saved_figures_mp4_and_svg_only.zip ./saved_figures**/**/*.svg ./saved_figures**/**/*.mp4  -x ./saved_figures**/**\step_** 

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

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 13

plt.rcParams['xtick.major.pad']='0'
plt.rcParams['ytick.major.pad']='0'
plt.rcParams['lines.linewidth']=2

COLOR = grey_darker
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR

filepaths = glob.glob("./saved_figures_copy/*/" + "*.p")
print(filepaths)


# For each picture in 
for pickle_file in filepaths:
    png_filepath = pickle_file.split('tracking_results_ep_')[0] + pickle_file.split('tracking_results_ep_')[1].split('.')[0] + "/"
    png_files = glob.glob(png_filepath + "step_*.png")

    [P_L_, P_R_, P_L_S_, P_R_S_, P_R_G_, P_L_G_, R_, R_S_, VL_, VR_] = pickle.load( open( pickle_file, "rb" ) )

    for png_file in png_files:

        regenerated_png_file = png_file.split('step_')[0] + "regenerated_step_" + png_file.split('step_')[1].split('_')[0]

        # Only load pickle data until that length
        step_idx = int(png_file.split('step_')[1].split('.')[0])

        P_L = np.array(P_L_[:step_idx])
        P_L_S = np.array(P_L_S_[:step_idx])
        P_L_G = np.array(P_L_G_)

        P_R = np.array(P_R_[:step_idx])
        P_R_S = np.array(P_R_S_[:step_idx])
        P_R_G = np.array(P_R_G_)
        VL=abs(VL_)
        VR=abs(VR_)

        fig, ax = plt.subplots(1,1, figsize=(9, 5), dpi=100)


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

        ax.set_title("Target and actuated pressures relative to atmosphere, \nV_L={:.01f}, \
            V_R={:.01f} (multiples of chamber volume)".format(VL, VR))
        ax.set_xlabel("Timesteps", fontsize=16)
        ax.set_ylabel("kPa", fontsize=16)

        ax.legend(["P_L observed real", "P_L goals", "P_R observed real", "P_R goals"])
        fig.savefig('./scripts/temp.png')
    
        # Rewrite tracking results over bottom half of image 
        # For each png file,
        plt.close()
        fig = plt.figure(frameon=False, figsize=(9, 9), dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        whole_image = plt.imread(png_file)
        tracking_results = plt.imread('./scripts/temp.png')
        tracking_results = tracking_results[:,:,:-1]
        # print(np.shape(whole_image), np.shape(tracking_results))
        whole_image[370:,:,:] = tracking_results
        plt.imshow(whole_image)
        fig.savefig(regenerated_png_file, format='png', dpi=100)
        plt.close()
        # plt.show()
    
    png_file_query = png_filepath+"regenerated_step_%03d.png"
    mp4_file_path = pickle_file.split('.p')[0] + ".mp4"
    # print(png_file_query)

    os.system(f"ffmpeg -r 3 -i {png_file_query} -vcodec mpeg4 -y {mp4_file_path}")
