#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
from cv2 import normalize
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from plyfile import *
import glob
from kitti360scripts.helpers.project import Camera, CameraPerspective
'''
3D render 2D image, 3D 投影到2D图像并画出
整个的坐标转换，从世界坐标系转换到图像坐标系
拥有的矩阵包括: 相机内参矩阵（intrinsic matrix，4x4,只需要3x3的部分）, 相机矫正矩阵（recification matrix 3x3）, cam2pose 矩阵（摄像机坐标到GPS/IMU坐标的转换矩阵）
输入：3D点的世界坐标系（xyz）以及颜色信息（rgb），经过坐标系转换得到对应的像素坐标信息（u，v，depth深度）

整个坐标的过程为：
得到GPU/IMU坐标到世界坐标系的刚体变换矩阵


输出4x4矩阵

得到current camera pose，从上方cam2world函数获得相机外参矩阵的R（旋转），T（平移），
函数cam2world的[:3,:3]是R（3X3），[:3,3]是T(3X1)

得到R和T之后，将世界坐标系投影到相机坐标系（world2cam），世界坐标系的点points=R.t*（世界坐标系（points）-T）（3X3 * （3X1-3X1））

得到相机坐标的点之后，将相机坐标系投影到图像坐标系中，用内参矩阵（Intrinsic*相机坐标系的点），得到的最后一行是深度信息depth
0为u, 1为v，2为depth


问题：首先，在代码中实现维度的变换，其次，什么时候应该inverse什么时候不需要，还是不理解
'''
#Main path
KITTI360Path="/media/ren/Seagate/KITTI360_DATASET/"

print(KITTI360Path)

# Define PLY types
# ply_dtypes = dict([
#     (b'int8', 'i1'),
#     (b'char', 'i1'),
#     (b'uint8', 'u1'),
#     (b'uchar', 'u1'),
#     (b'int16', 'i2'),
#     (b'short', 'i2'),
#     (b'uint16', 'u2'),
#     (b'ushort', 'u2'),
#     (b'int32', 'i4'),
#     (b'int', 'i4'),
#     (b'uint32', 'u4'),
#     (b'uint', 'u4'),
#     (b'float32', 'f4'),
#     (b'float', 'f4'),
#     (b'float64', 'f8'),
#     (b'double', 'f8')
# ])


#image pixel 
width=1408 #width
height=376 #height
#template pixel
tempWl=100 #width
tempWr=1300
tempHu=10 #height
tempHd=350
#index of frame
frameId=0
#maximum and minimum value within matrix
max_v=0
min_v=0
# cam2pose=[] #unrecified camera to GPU/IMU matrix 
poses=[] #GPU/IMU to world transform matrix
R_rectMat=np.eye(4)#recification matrix
pcMat=[]#point cloud matrix
pcMatC=[]#point cloud color matrix
# IntrinsicMat=[] #intrinsic matrix
# cam2poseMat=[]#transformation matrix from camera frame to GPS frame
# plymat=[]#point cloud matrix

#rendered image matrix 
# imageMat=np.zeros((tempHd-tempHu+1),(tempWr-tempWl+1))#,np.array)
# imageMat=np.zeros(1200,340)
# h,w=imageMat.shape
# imgMat = cv2.CreateMat(h, w, cv2.CV_64FC4)

# imgMat = cv2.fromarray(imageMat)

#perspective file
perspective=os.path.join(KITTI360Path,"calibration/perspective.txt")
#calib_cam2pose file
calib_cam2pose=os.path.join(KITTI360Path,"calibration/calib_cam_to_pose.txt")


#intrinsic matrix and recification matrix load from perspective
def load_intrinsics(intrinsic_file):
    #This function is camera Intrinsic matrix, P_rect_00 and R_rect_00
    with open(intrinsic_file) as f:
        intrinsics = f.read().splitlines()
    for line in intrinsics:
        line = line.split(' ')
        #3x4 intrinsic matrix from perspective.txt
        if line[0] == 'P_rect_00:' :
            IntrinsicMat = [float(x) for x in line[1:]]
            IntrinsicMat = np.reshape(IntrinsicMat, [3,4])
 
        #rectification matrix: eye(4) with the left up 3x3 filled with numbers from perspective.txt
        elif line[0] == 'R_rect_00:':
            # R_rect = np.eye(4) 
            R_rectMat[:3,:3] = np.array([float(x) for x in line[1:]]).reshape(3,3)

            
    print("intrinsic",IntrinsicMat)
    print("R_rect",R_rectMat)
    # print("cam2pose calib",calic2p)
    return IntrinsicMat, R_rectMat,# calic2p
#read cam2poseMat
def load_cam2pose(calib_cam2pose):
    #transformation matrix from camera frame to GPS frame
    with open(calib_cam2pose) as f:
        image00data = f.read().splitlines()
    for line in image00data:
        line = line.split(' ')

        if line[0] == 'image_00:' :    
            cam2poseMat=np.array([float(x) for x in line[1:]]).reshape(3,4)
            cam2poseMat=np.concatenate((cam2poseMat, np.array([0.,0.,0.,1.]).reshape(1,4)))

    print("cam2pose calib",cam2poseMat)
    return cam2poseMat
frame=[]




def load_poses(seq):
    #This function is to open poses file 
    #cam2poses file
    posespath=os.path.join(KITTI360Path,'data_poses/2013_05_28_drive_00%s_sync/'%seq,"cam0_to_world.txt")
    # print('poses',posespath)
    
    with open(posespath) as f:
        posesdata = f.read().splitlines()
    for line in posesdata:
        line = line.split(' ')
        # print('line',line)
        # for i in line:
        if '' in line:
            line.remove('')
        # print('line',line)
        # print('line',line[1:])
        frame.append(int(line[0]))
        # print('frame',line[0])
        # print('len line',len(line))  
        pose=np.array([float(x) for x in line[1:]]).reshape(4,4)
        R=pose[:3,:3]
        T=pose[:3,3]
        
        points=np.matmul(R.T,(points[frame]-T))
        points_proj=np.matmul(IntrinsicMat[:3,3],points)
        
        depth=points_proj[2,:]
        
        depth[depth==0]= -1e-6
        
        #normalized image coordinates
        u = np.round(points_proj[0,:]/np.abs(depth)).astype(np.int)
        v = np.round(points_proj[1,:]/np.abs(depth)).astype(np.int)
        
        
        # print('pose',pose)
        # pose=np.array(np.concatenate((pose, np.array([0.,0.,0.,1.]).reshape(1,4))))
        # print('pose',pose)
        
        # poses.appened(pose)
    print("poses",poses)
    return u,v,depth


#read matrix from files
IntrinsicMat,R_rectMat=load_intrinsics(perspective)#read intrinsic matrix and rectification matrix
cam2poseMat=load_cam2pose(calib_cam2pose)#read cam2poseMat
# print('Matrix',IntrinsicMat,R_rectMat)
rectinv=np.linalg.inv(R_rectMat)#inverse of rectification matrix
print('inv',rectinv)
frame,poseMat=load_poses("00")
# print("frame",frame,len(frame))

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
        print(filelist[i],i)
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
        pcMat=np.concatenate([pcMat,pc],axis=1)
            
        

            
        # pcMat=np.concatenate([pcMat,pc],axis=0)
        pcMatC=np.concatenate([pcMatC,pc_color],axis=1)
    # print(pc.shape)
    # pcMat
    return pcMat,pcMatC

pcMat,pcMatC=read_ply("00","static")
print('pcMat',pcMat,pcMat.shape)
# print(pcMatC,pcMatC.shape)
'''
Now we have R_rect, P_rect and cam2pose matrix from txt file
and points as matrix in ply file, now the transformation from world to image should be done
'''

#frame by frame management

# def posesPath(seq):
#     listdir=os.listdir(os.path.join(KITTI360Path,"data_2d_raw/2013_05_28_drive_00%s_sync"%seq,"image_00/data_rect"))
#     # print(listdir)
#     print(os.path.join(KITTI360Path,"data_2d_raw/2013_05_28_drive_00%s_sync"%seq,"image_00/data_rect"))
#     file_nums = len(listdir)-1#sum([os.path.isdir(listx) for listx in listdir]) 
#     return file_nums

def framePath(seq):
    listdir=os.listdir(os.path.join(KITTI360Path,"data_2d_raw/2013_05_28_drive_00%s_sync"%seq,"image_00/data_rect"))
    # print(listdir)
    print(os.path.join(KITTI360Path,"data_2d_raw/2013_05_28_drive_00%s_sync"%seq,"image_00/data_rect"))
    file_nums = len(listdir)-1#sum([os.path.isdir(listx) for listx in listdir]) 
    return file_nums
nums=framePath("00")

# def framePath(seq):
#     listdir=os.listdir(os.path.join(KITTI360Path,"data_2d_raw/2013_05_28_drive_00%s_sync"%seq,"image_00/data_rect"))
#     # print(listdir)
#     print(os.path.join(KITTI360Path,"data_2d_raw/2013_05_28_drive_00%s_sync"%seq,"image_00/data_rect"))
#     file_nums = len(listdir)-1#sum([os.path.isdir(listx) for listx in listdir]) 
#     return file_nums


# print(l)

posespath=os.path.join(KITTI360Path,'data_poses/2013_05_28_drive_0000_sync/',"poses.txt")
poses = np.loadtxt(posespath)
cam2world = {}


(u,v),depth=Camera.project_vertices(pcMat,frameId=0)


#converts points from world frame to camera frame， input is 3D array, world frame is a 3D array, camera frame is a 2D array 
def world2cam(points, R, T, inverse=False):
        assert (points.ndim==R.ndim)
        assert (T.ndim==R.ndim or T.ndim==(R.ndim-1)) 
        ndim=R.ndim
        #if array's dimension is 2, add 1 dimension at the beginning
        if ndim==2:
            R = np.expand_dims(R, 0) 
            T = np.reshape(T, [1, -1, 3]) #add 1 dimension at the beginning and reshape T into Nx3 shape
            points = np.expand_dims(points, 0)
        #if not inverse, T and R are translation and rotation that transforms points from world to camera frame
        #if inverse, T and R transforms points from camera to world frame
        #transpose of R is the inverse of R. Since the additional dimension is at the beginning, we can just ignore it,
        #and switch dimension 1 and 2, which is transpose(0,2,1). 
        #However, I don't know why it transpose the point coordinates after matmul(R,points)?????????????????
        if not inverse:
            points = np.matmul(R, points.transpose(0,2,1)).transpose(0,2,1) + T
        else:
            points = np.matmul(R.transpose(0,2,1), (points - T).transpose(0,2,1))

        #if input array is 2d, get rid of the dimension added, so the output array is also 2d
        if ndim==2:
            points = points[0]

        return points

def cam2image(points):
        #if points array has 2 dimensions, add one dimension at the beginning
        ndim = points.ndim
        if ndim == 2:
            points = np.expand_dims(points, 0)
        # matmul(intrinsic matrix, points) = position on image, points are camera coordinates
        points_proj = np.matmul(IntrinsicMat[:3,:3].reshape([1,3,3]), points)
        depth = points_proj[:,2,:]
        depth[depth==0] = -1e-6
        # normalised image coordinates
        u = np.round(points_proj[:,0,:]/np.abs(depth)).astype(np.int)
        v = np.round(points_proj[:,1,:]/np.abs(depth)).astype(np.int)
        
        #if input array is 2d, drop the added dimension so the output array is also 2d
        if ndim==2:
            u = u[0]; v=v[0]; depth=depth[0]
        return u, v, depth


#Plot points on images for rendering 画图，将2D点画到画布上

# for i in range (nums):
    
    
    
#     poseMat=[]
#     # print('poseMat',poseMat.shape)
#     # print('cam2poseMat',cam2poseMat.shape)
#     print('rectinv',rectinv)
#     cam2world=np.matmul(np.matmul(poseMat,cam2poseMat),rectinv)
#     R=cam2world[:3,:3]
#     T=cam2world[:3,3]
#     #point from 
#     pointW=np.matmul(R.transpose(0,2,1), (poseMat - T).transpose(0,2,1))
    
#     pointC=np.matmul(np.transpose(R),pointW)
    
#     points_proj=np.matmul(IntrinsicMat,pointW)
    
#     d = points_proj[:,2,:]
#     d[d==0] = -1e-6
#     # normalised image coordinates
#     u = np.round(points_proj[:,0,:]/np.abs(d)).astype(np.int)
#     v = np.round(points_proj[:,1,:]/np.abs(d)).astype(np.int)

# def PLYfileload(plyfile,ext):

#         line = []
#         vertex_properties = []
#         num_points = None
#         num_faces = None
#         current_element = None 
       
#         while b'end_header' not in line and line != b'':
#             line = plyfile.readline()

#         # Find point element
#             if b'element vertex' in line:
#                 current_element = 'vertex'
#                 line = line.split()
#                 num_points = int(line[2])

#             elif b'element face' in line:
#                 current_element = 'face'
#                 line = line.split()
#                 num_faces = int(line[2])

#             elif b'property' in line:
#                 if current_element == 'vertex':
#                     line = line.split()
#                     vertex_properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
#                 elif current_element == 'vertex':
#                     if not line.startswith('property list uchar int'):
#                         raise ValueError('Unsupported faces property : ' + line)
#         return num_points, num_faces, vertex_properties

# plymat=PLYfileload('00')

# def read_ply(filename):
#     """ read XYZ point cloud from filename PLY file """
#     plydata = PlyData.read(filename)
#     print(plydata)
#     pc = plydata['vertex'].data
    
#     pc_array = np.array([[x, y, z] for x,y,z in pc])
    
#     return pc_array

# pclfile=os.path.join(KITTI360Path,'data_3d_semantics/2013_05_28_drive_0000_sync/static/000002_000385.ply')

# plymat=read_ply(pclfile)

# def Loadfile(seq):
#         print('seq',seq)    
#         if seq.isnumeric:
#             print('sequence number type right, load sequence')
#         cam2pose_file = os.path.join(KITTI360Path, "data_poses/2013_05_28_drive_00%2s_sync"%seq,"cam0_to_world.txt")#'Calcalibration/calib_cam_to_pose.txt'
#         poses_file = os.path.join(KITTI360Path, "data_poses/2013_05_28_drive_00%2s_sync"%seq,"poses.txt")
        
#         cam2pose=np.loadtxt(cam2pose_file)
#         # print(cam2pose.shape)
#         frames=cam2pose[:,0]
#         print(frames.shape)
#         poses=np.loadtxt(poses_file)
#         # print(poses.shape)
#         poses = np.reshape(poses[:,1:],[-1,3,4]) 
#         # print(poses.shape)
#         IntrinsicMat,R_rectMat=load_intrinsics(perspective)
#         calicam2po=load_cam2pose(calib_cam2pose)
#         R_inv=np.linalg.inv(R_rectMat)
#         print("inverse R",R_inv)
        
#         return None
        

#  def PLYfileload(seq):
#     plydata = PlyData.read(os.path.join(KITTI360Path,'data_3d_semantics/train/2013_05_28_drive_00%s_sync/static'%seq))
#     with open(plydata, 'rb') as f:
#         plydata = PlyData.read(f)
        
#         xlist = plydata['vertex']['x']
#         ylist = plydata['vertex']['y']
#         zlist = plydata['vertex']['z']
        
        
#     return xlist,ylist,zlist


    #     with open(intrinsic_file) as f:
    #         intrinsics = f.read().splitlines()
    #     for line in intrinsics:
    #         line = line.split(' ')
    # #3x4 intrinsic matrix from perspective.txt
    #     if line[0] == 'P_rect_00:' :
    #         Intrinsic = [float(x) for x in line[1:]]
    #         Intrinsic = np.reshape(Intrinsic, [3,4])

    # #rectification matrix: eye(4) with the left up 3x3 filled with numbers from perspective.txt
    #     elif line[0] == 'R_rect_00:':
    #     # R_rect = np.eye(4) 
    #         R_rect[:3,:3] = np.array([float(x) for x in line[1:]]).reshape(3,3)
    #image width and height, first number width, second number height
        
        # line = line[:-1]
        # print('line',line)
        
        # print(cam2pose_file,poses_file)
        # print(cam2pose)
        # print(poses)
        # return cam2pose,poses 
   
# Intrinsic, R_rect=load_intrinsics() 
# sequence00=Loadfile('00')
        # calib_dir = os.path.join(root_dir, 'calibration')
        # self.pose_file = os.path.join(pose_dir, "poses.txt")
        # self.intrinsic_file = os.path.join(calib_dir, 'perspective.txt')
        # fileCameraToPose = os.path.join(calib_dir, 'calib_cam_to_pose.txt')
        # self.camToPose = loadCalibrationCameraToPose(fileCameraToPose)['image_%02d' % cam_id]
# (x, y, z, intensity)
# def pointcloudread(PlyfilePath):
    
#     """ read XYZ point cloud from filename PLY file """
#     PlyfilePath=()
#     plydata = plydata.read(PlyfilePath)
#     pc = plydata['vertex'].data
#     pc_array = np.array([[x, y, z] for x,y,z in pc])
#     return pc_array 



# def pointcloud2image(point_cloud):
#     x=pointcloudread(point_cloud)[0]
#     y=pointcloudread(point_cloud)[1]
#     z=pointcloudread(point_cloud)[2]
#     # grid_size = np.array([2 * x_range / x_size, 2 * y_range / y_size])
#     # image_size = np.array([x_size, y_size])
#     return x,y,z
# x,y,z=pointcloud2image()
# # [0, 2*range)
    # shifted_coord = point_cloud[:, :2] + np.array([x_range, y_range])

# image index
    # index = np.floor(shifted_coord / grid_size).astype(np.int)

# choose illegal index
#     bound_x = np.logical_and(index[:, 0] >= 0, index[:, 0] < image_size[0])
#     bound_y = np.logical_and(index[:, 1] >= 0, index[:, 1] < image_size[1])
#     bound_box = np.logical_and(bound_x, bound_y)
#     index = index[bound_box]

# # show image
#     image = np.zeros((640, 640), dtype=np.uint8)
#     image[index[:, 0], index[:, 1]] = 255
#     res = Image.fromarray(image)

# # rgb = Image.merge('RGB', (res, res, res))
#     res.show()
    
# class Camera:
#     def __init__(self):
        
#         # load intrinsics
#         self.load_intrinsics(self.intrinsic_file)

#         # load poses
#         poses = np.loadtxt(self.pose_file)
#         frames = poses[:,0]
#         poses = np.reshape(poses[:,1:],[-1,3,4])
#         self.cam2world = {}
#         self.frames = frames
#         for frame, pose in zip(frames, poses): 
#             pose = np.concatenate((pose, np.array([0.,0.,0.,1.]).reshape(1,4)))
#             # consider the rectification for perspective cameras
#             if self.cam_id==0 or self.cam_id==1:
#                 self.cam2world[frame] = np.matmul(np.matmul(pose, self.camToPose),
#                                                   np.linalg.inv(self.R_rect))
#             # fisheye cameras
#             elif self.cam_id==2 or self.cam_id==3:
#                 self.cam2world[frame] = np.matmul(pose, self.camToPose)
#             else:
#                 raise RuntimeError('Unknown Camera ID!')


#     def world2cam(self, points, R, T, inverse=False):
#         assert (points.ndim==R.ndim)
#         assert (T.ndim==R.ndim or T.ndim==(R.ndim-1)) 
#         ndim=R.ndim
#         if ndim==2:
#             R = np.expand_dims(R, 0) 
#             T = np.reshape(T, [1, -1, 3])
#             points = np.expand_dims(points, 0)
#         if not inverse:
#             points = np.matmul(R, points.transpose(0,2,1)).transpose(0,2,1) + T
#         else:
#             points = np.matmul(R.transpose(0,2,1), (points - T).transpose(0,2,1))

#         if ndim==2:
#             points = points[0]

#         return points

#     def cam2image(self, points): 
#         raise NotImplementedError

#     def load_intrinsics(self, intrinsic_file):
#         raise NotImplementedError
    
#     def project_vertices(self, vertices, frameId, inverse=True):

#         # current camera pose
#         curr_pose = self.cam2world[frameId]
#         T = curr_pose[:3,  3]
#         R = curr_pose[:3, :3]

#         # convert points from world coordinate to local coordinate 
#         points_local = self.world2cam(vertices, R, T, inverse)

#         # perspective projection
#         u,v,depth = self.cam2image(points_local)

#         return (u,v), depth 

#     def __call__(self, obj3d, frameId):

#         vertices = obj3d.vertices

#         uv, depth = self.project_vertices(vertices, frameId)

#         obj3d.vertices_proj = uv
#         obj3d.vertices_depth = depth 
#         obj3d.generateMeshes()