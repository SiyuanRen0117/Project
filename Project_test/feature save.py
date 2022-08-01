#-*-coding:utf-8-*-
import cv2
import numpy as np
from sklearn.cluster import KMeans
import sys
import os

path = sys.path[0] + os.sep

def feature_extract(images_folder, draw_folder):
    featureSum = 0
    for filename in os.listdir(images_folder):
        if '.png' in filename:
            filepath = images_folder + filename
            drawpath = draw_folder + filename  
        else:
            continue  
        print(filepath)
        img = cv2.imread(filepath)
        
        filename = filepath.split(os.sep)[-1].split('.')[0]
        print(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # set Hessian threshold
        detector = cv2.xfeatures2d.SURF_create(2000)
        # find keypoints and descriptors directly
        kps, des = detector.detectAndCompute(gray, None)
        img = cv2.drawKeypoints(image=img, outImage=img, keypoints=kps, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(255, 0, 0))
        feature_name = images_folder + 'features%s%s.feature'%(os.sep, filename)
        try:
            np.savetxt(feature_name, des, fmt='%.5e') # save feature vector
            # feature = np.loadtxt(feature_folder + filename) # load feature vector
            # cv2.imwrite(drawpath, img) # plot the features
        except:
            continue
        featureSum += len(kps)
    print (featureSum)

def main():
    images_folder = path + 'images' + os.sep
    draw_folder = path + 'results' + os.sep + 'drawImages' + os.sep
    feature_extract(images_folder, draw_folder)

featureSum=feature_extract('/media/ren/Seagate/KITTI360_DATASET/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/', '/media/ren/Seagate/KITTI360_DATASET/data_2d_raw/2013_05_28_drive_0000_sync/image_00/SURF_feature_detection/')