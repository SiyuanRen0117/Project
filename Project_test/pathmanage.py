#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from plyfile import *



#Main path
KITTI360Path="/media/ren/Seagate/KITTI360_DATASET/"

def framePath(seq):
    path=os.path.join(KITTI360Path,"data_poses/2013_05_28_drive_00%s_sync"%seq,"poses.txt")
    f=open(path,'r')
    f.readlines()
    print(f)
    # print(os.path.join(KITTI360Path,"data_2d_raw/2013_05_28_drive_00%s_sync"%seq,"image_00/data_rect"))
    # file_nums = len(listdir)-1#sum([os.path.isdir(listx) for listx in listdir]) 
    return f

path=framePath("00")