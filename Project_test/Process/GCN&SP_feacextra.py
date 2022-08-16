import numpy as np
import cv2
import os
from PIL import Image
from matplotlib import pyplot as plt
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Main path

KITTI360Path="/media/ren/EXTERNAL_USB/KITTI360_DATASET/"

print(KITTI360Path)
# NoneType=type(None)


# img1=Image.open(os.path.join(KITTI360Path,'data_2d_raw/2013_05_28_drive_0007_sync/image_00/data_rect/0000000003.png'))
# img1=Image.open(os.path.join(KITTI360Path,'data_2d_raw/2013_05_28_drive_0005_sync/image_00/data_rect/0000000001.png'))
img1=Image.open(os.path.join(KITTI360Path,'Test_data/00000.jpg'))
# img2=Image.open(os.path.join(KITTI360Path,'data_rendered_image/2013_05_28_drive_0005_sync','frame_02.jpg'))
# img2=Image.open(os.path.join(KITTI360Path,'data_rendered_image/2013_05_28_drive_0005_sync','frame_00.png'))
# img2=Image.open(os.path.join(KITTI360Path,'data_rendered_image/2013_05_28_drive_0005_sync','05_frame_00.png'))
img2=Image.open(os.path.join(KITTI360Path,'Test_data/00000-simulation.jpg'))
x=img1.width
y=img1.height

img1 = cv2.imread(os.path.join(KITTI360Path,'Test_data/00000.jpg'))
# img1 = cv2.imread(os.path.join(KITTI360Path,'data_2d_raw/2013_05_28_drive_0005_sync/image_00/data_rect/0000000001.png'))
# img2 = cv2.imread(os.path.join(KITTI360Path,'frame.jpg'))

# img2 = cv2.imread(os.path.join(KITTI360Path,'data_rendered_image/2013_05_28_drive_0005_sync','05_frame_00.png'))
# img2 = cv2.imread(os.path.join(KITTI360Path,'data_rendered_image/2013_05_28_drive_0005_sync','capture1_1.jpg'))
img2 = cv2.imread(os.path.join(KITTI360Path,'Test_data/00000-simulation.jpg'))
# img1=cv2.resize(img1,(x,y))
img2=cv2.resize(img2,(x,y))
def GCNExtractor(img):
    img_width = 640
    img_height = 480
    border = 16
    dist_thresh = 8

    ratio_width = float(img1.cols) / float(img_width)
    ratio_height = float(img1.rows) / float(img_height)

    img1=cv2.resize(img1,(img_width,img_height))
    img2=cv2.resize(img2,(img_width,img_height))


