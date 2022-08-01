#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from plyfile import *
# sys.append("/home/ren/kitti360Scripts/kitti360scripts/helpers/ply.py")

#Main path
KITTI360Path="/media/ren/Seagate/KITTI360_DATASET/"
# # [1000.8657 1000.8753 1000.9065 ...  817.023   817.055   817.602 ]
# #  [3820.2976 3820.3816 3820.6013 ... 3156.94   3157.05   3156.37  ]
# #  [ 116.2225  116.193   116.188  ...  115.263   115.259   114.905 ]

# print(KITTI360Path)

# PLpoints=[]

# def read_ply(filename):
#     """ read XYZ point cloud from filename PLY file """
#     plydata = PlyData.read(filename)
#     print('plydata',plydata)
#     pc_x = np.expand_dims(np.array(plydata['vertex']['x']),1)#.reshape()
#     pc_y = np.expand_dims(np.array(plydata['vertex']['y']),1)
#     pc_z = np.expand_dims(np.array(plydata['vertex']['z']),1)
#     pc_r = np.expand_dims(np.array(plydata['vertex']['red']),1)#.reshape()
#     pc_g = np.expand_dims(np.array(plydata['vertex']['green']),1)
#     pc_b = np.expand_dims(np.array(plydata['vertex']['blue']),1)
    
#     print(pc_x.shape,pc_y.shape,pc_z.shape)
#     pc=np.concatenate((pc_x,pc_y,pc_z),1)
#     pc_color=np.concatenate((pc_r,pc_g,pc_b),1)
#     print(pc_color)
#     # pc=list(pc)
#     # print('pc',pc[:10])
#     # aa = [(1,2,3), (1,2,3)]
#     # print("test ", np.array([*aa]))
#     # pc_array=np.array([*pc])#.reshape(pc.shape,1)
#     # print('pc',pc_array[:10])
#     # pc_array.reshape(pc_array.shape,1)
#     # print(pc.shape)
#     # print('pc',pc[0])
#     # print('0',)
#     # print('pc',pc[0][1])
#     # for i in range (2):
#     #     ply=pc[i]
#     #     for j in range(pc.shape):
#     #         pcl=[ply[0],ply[1],ply[2]].reshape(1,3)
            
#     #         PLpoints.append(pcl)
#     print(pc.shape)
#     print('color',pc_color)
#     return pc,pc_color

# pclfile=os.path.join(KITTI360Path,'data_3d_semantics/2013_05_28_drive_0000_sync/static/000002_000385.ply')



#ply read
def read_ply(seq,str):
    """ read XYZ point cloud from filename PLY file """
    pc_path=os.path.join(KITTI360Path,'data_3d_semantics/2013_05_28_drive_00%s_sync/'%seq,str)
    # print(pc_path)
    filelist = os.listdir(pc_path)
    filelist=sorted(filelist)
    # print('file',filelist)
    # print(len(filelist))
    for i in range(len(filelist)):
        # print(filelist[i],i)
        plydata = PlyData.read(os.path.join(pc_path,filelist[i]))
        # print('plydata',plydata)
        pc_x = np.expand_dims(np.array(plydata['vertex']['x']),1)
        pc_y = np.expand_dims(np.array(plydata['vertex']['y']),1)
        pc_z = np.expand_dims(np.array(plydata['vertex']['z']),1)
        pc_r = np.expand_dims(np.array(plydata['vertex']['red']),1)#.reshape()
        pc_g = np.expand_dims(np.array(plydata['vertex']['green']),1)
        pc_b = np.expand_dims(np.array(plydata['vertex']['blue']),1)
        # print(pc_x.shape,pc_y.shape,pc_z.shape)
        pc=np.concatenate((pc_x,pc_y,pc_z),1)
        # print('pcc',pc)
        pc=np.transpose(np.concatenate((pc_x,pc_y,pc_z),1))
        # print('pc',pc)
        pc_color=np.transpose(np.concatenate((pc_r,pc_g,pc_b),1))
        
        if i == 0:
            pc_0=pc
            pc_c0=pc_color
            pcMat=pc_0
            pcMatC=pc_c0
            # print('pc_0',pc_0)
            # print('pc_c0',pc_c0)
        # print(pc_0)
        pcMat=np.array(np.concatenate([pcMat,pc],axis=1))
            
            
        # pcMat=np.concatenate([pcMat,pc],axis=0)
        pcMat_RGB=np.array(np.concatenate([pcMatC,pc_color],axis=1))
    # print(pc.shape)
    # pcMat
    return pcMat,pcMat_RGB

pcMat,pcMatC=read_ply("00","static")

print('pcMat',pcMat,pcMat.shape)
R=np.random.rand(3,3)
T=np.random.rand(3,1)

print(R,T)
print('pcMat',pcMat)
points_local=np.matmul(R.T,(pcMat[:,:]-T[:,:]))
print('local',points_local)
# print('T shape',np.repeat(np.expand_dims(T,axis=1),pcMat.shape[1],axis=1).shape)
# print((np.expand_dims(pcMat[:,:],axis=1)-np.repeat(np.expand_dims(T,axis=1),pcMat.shape[1],axis=1)).shape)
# p=np.matmul(R.T,np.repeat(np.expand_dims(T,axis=1),pcMat.shape[1],axis=1)
# print('mat shape',((np.expand_dims(pcMat,axis=2)).shape))

# print((np.repeat(T,pcMat.shape[1],axis=1)))
# points_local=np.matmul(R.T,(pcMat[:,:]-np.repeat(T,pcMat.shape[1],axis=1)))
# pc,pc_color=read_ply(pclfile)
# print()