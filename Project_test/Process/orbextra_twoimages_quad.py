import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from plyfile import *


#Main path

KITTI360Path="/media/ren/EXTERNAL_USB/KITTI360_DATASET/"

print(KITTI360Path)

# img1=Image.open('/media/ren/EXTERNAL_USB/render_test/render_test/raw/220403_151907_01_01.jpg')



img2=Image.open('/media/ren/EXTERNAL_USB/render_test/render_test/render/624.png')
x=img2.width
y=img2.height

# img1=cv2.imread


img1 = cv2.imread(os.path.join(KITTI360Path,'/media/ren/EXTERNAL_USB/render_test/render_test/raw/220403_151907_01_01.jpg'))
img2 = cv2.imread(os.path.join(KITTI360Path,'/media/ren/EXTERNAL_USB/render_test/render_test/render/857.png'))
img1=cv2.resize(img1,(x,y))
img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

# Image.open(os.path.join(KITTI360Path,'/media/ren/EXTERNAL_USB/render_test/render_test/raw/220403_151907_01_01.jpg'))
img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
# print(kp1)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

matches = bf.match(des1, des2)

matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[: 100], None, flags=2)
plt.imshow(img3), plt.show()
