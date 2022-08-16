
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import time

import matplotlib.pyplot as plt
from plyfile import *


#Main path

KITTI360Path="/media/ren/EXTERNAL_USB/KITTI360_DATASET/"

print(KITTI360Path)



img1=Image.open(os.path.join(KITTI360Path,'data_2d_raw/2013_05_28_drive_0007_sync/image_00/data_rect/0000000000.png'))
img2=Image.open(os.path.join(KITTI360Path,'data_rendered_image/2013_05_28_drive_0007_sync','capture3.jpg'))
x=img1.width
y=img1.height
img1 = cv2.imread(os.path.join(KITTI360Path,'data_2d_raw/2013_05_28_drive_0007_sync/image_00/data_rect/0000000000.png'))
# img2 = cv2.imread(os.path.join(KITTI360Path,'frame.jpg'))
img2 = cv2.imread(os.path.join(KITTI360Path,'data_rendered_image/2013_05_28_drive_0007_sync','capture3.jpg'))
# img1=cv2.resize(img1,(x,y))
img2=cv2.resize(img2,(x,y))
# img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
# img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

orb = cv2.ORB_create(nfeatures=2000)
print('orb',orb)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
# print(len(kp1))
print(des1.shape)
print(des2.shape)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
print("bf",bf)
# np.savetxt('0007_raw_keypoint.txt',kp1,fmt='%d')
# np.savetxt('0007_rendered_keypoint.txt',kp2,fmt='%d')
# np.savetxt('0007_raw_descriptor.txt',des1,fmt='%d')
# np.savetxt('0007_rendered_descriptor.txt',des2,fmt='%d')
matches = bf.match(des1, des2)

matches = sorted(matches, key = lambda x:x.distance)

print("matches",matches)
# img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[: 2000], None, flags=2)
# plt.imshow(img3), plt.show()


# Apply ratio test
#ratio test, find the closest point B and second point C from point A
#matching when B/C is smaller than threshold 0.75
#the ideal matching distance is 0 if the matching are corrsponsed.

good = []
pts1 = []
pts2 = []
print('distance',matches[0].distance,'matching number',len(matches))
for m in matches:
    # print('m before',m.distance)
    if m.distance < max(2 * matches[0].distance, 30): #distance choose, 
        # print('m',m.distance)
        good.append([m])
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)
time1=time.time()
pts1 = np.asarray(pts1)
print('pts1',pts1.shape)

pts2 = np.asarray(pts2)
print('pts2',pts2.shape)
# #计算基础矩阵 1ms
good_F, status = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3, 0.99,100)
time2=time.time()
print('len: ', len(good), np.sum(status),time2-time1)	#内点数

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.imshow("image1", img3)
cv2.waitKey(0)  
# cv2.destroyAllWindows()  
# print(matches)

RANSAC(img1, img2, kp1, kp2, matches)
# img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[: 2000], None, flags=2)
# plt.imshow(img3), plt.show()
