import numpy as np
import cv2
import os
from PIL import Image
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

detector = cv2.ORB_create(nfeatures=2000)
flann_params= dict(algorithm = 6,table_number = 6, # 12 
                   key_size = 12, # 20
                   multi_probe_level = 1) #2

matcher = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

kp1,desc1 = detector.detectAndCompute(img1,None)

kp2,desc2 = detector.detectAndCompute(img2,None)

raw_matches = matcher.knnMatch(desc1, desc2,2) #2

good = []

for m,n in raw_matches:
    print('m',m.distance)
    print('n',n.distance)
    if m.distance < 0.7*n.distance:

        good.append(m)

        if len(good)>10:

            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)

            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

            matchesMask = mask.ravel().tolist()

    else:

        print("Not enough matches are found - %d/%d" % (len(good),10))

        matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                singlePointColor = (0,0,255), matchesMask = matchesMask, flags = 2)# draw only inliers

vis = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

cv2.imshow("", vis)

cv2.imwrite("pic/face_brisk_bf_ransac_1519.jpg", vis)

cv2.waitKey()

cv2.destroyAllWindows()