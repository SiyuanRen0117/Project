import os

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from plyfile import *
import glob
import time
from kitti360scripts.helpers.project import Camera, CameraPerspective
from kitti360scripts.helpers.ply import *

# R=np.array([-0.938425,0.01471056,-0.345172],[ 0.345449,0.05439153,-0.93686],[0.004993,-0.9984115, -0.056123])

R=[[-0.938425,0.01471056,-0.345172],
   [ 0.345449,0.05439153,-0.93686],
   [0.004993,-0.9984115, -0.056123]]

print(R)
R=np.array(R)
trans=[[155.20732266],
       [96.64960742],
       [0.22340366]]


trans=np.array(trans)
mat=np.matmul(R.T,trans)
print(mat)

