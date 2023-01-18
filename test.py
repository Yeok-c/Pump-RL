import cv2
from resources.graphics import Graphics
import numpy as np


graphics = Graphics()
# background = make_alpha(background, 0)
A = cv2.imread('./saved_figures/0/0/step_001.png')
A = A[:450,:,:]
B = cv2.imread('./saved_figures/0/0/tracking_step_001.png')
# C = graphics._overlay_images(background, A, 0, 0)
C = np.concatenate((A,B), axis=0)
cv2.imshow('C', C)
cv2.waitKey(0)