import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from plyfile import *


#Main path

KITTI360Path="/media/ren/Seagate/KITTI360_DATASET/"

print(KITTI360Path)

img1 = cv2.imread(os.path.join(KITTI360Path,'data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000000.png'))
img2 = cv2.imread(os.path.join(KITTI360Path,'data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000000.png'))
img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

matches = bf.match(des1, des2)

matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[: 20], None, flags=2)
plt.imshow(img3), plt.show()
