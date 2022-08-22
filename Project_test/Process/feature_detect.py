from itertools import count
import numpy as np
import cv2
import os
from PIL import Image
from matplotlib import pyplot as plt


"""
Need a measurement of matching
"""

#Main path

KITTI360Path="/media/ren/EXTERNAL_USB/KITTI360_DATASET/"

print(KITTI360Path)
# NoneType=type(None)
def ORB(img):
    """
     ORB detector
    """
    orb = cv2.ORB_create(50000)
    """find keypoint, and calculate descriptor"""
    kp, des = orb.detectAndCompute(img, None)

    # plot keypoints
    img2 = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0), flags=0)

 
    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(img2[:, :, ::-1])
    plt.xticks([]), plt.yticks([])
    plt.show()
    return kp, des



def SIFT(img):
    """
     SIFT detector
    """
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # SIFT keypoint detector
    # sift instantiation 
    sift = cv2.xfeatures2d.SIFT_create()

    # sift detect and compute 
    kp, des = sift.detectAndCompute(img, None)
    
    # kp: information of keypoint, including position, scale, direction
    # des: keypoint descriptor, corresponds to a feature vector of 128 gradient information

    # image show
    cv2.drawKeypoints(img, kp, img, (255, 0, 0), flags=0)

    
    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(img[:, :, ::-1])
    plt.xticks([]), plt.yticks([])
    plt.show()
    return kp, des


def SURF(img):
    surf = cv2.xfeatures2d.SURF_create()
    kp, des = surf.detectAndCompute(img, None)
    cv2.drawKeypoints(img, kp, img, (255, 0, 0), flags=0)

    
    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(img[:, :, ::-1])
    plt.xticks([]), plt.yticks([])
    plt.show()
    return kp, des


"""
Feature matching method
(1). Brute Force matching
(2). FLAnn Matching
"""


def Brute(img1, img2, kp1, kp2, des1, des2, flag):
    """
    Brute Force matching
    :param img1: image 1
    :param img2: image 2
    :param kp1: keypoints of frame 1True
    :param kp2: keypoints of frame 2
    :param des1: descriptor of frame 1
    :param des2: descriptor of frame 2
    :return:
    """
    print("img1",len(kp1),len(kp2),len(des1),len(des2))
    # if type(des1)!=NoneType and type(des2)!=NoneType:
    if (flag == "SIFT" or flag == "SURF"):
        # SIFT
        bf = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=True)
        
    else:
        # ORB
        # bf = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=True)
        bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
    # print('des1',des1.shape)
    # print('des2',des2.shape)
    # ms = bf.knnMatch(des1, des2, k=2)
    ms = bf.match(des1, des2)
    # ms = sorted(ms, key=lambda x: x.distance)
    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, ms, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow("Matches", img3)
    # cv2.waitKey(0)
    return ms

# def Brute(img1, img2, kp1, kp2, des1, des2, flag="ORB"):
#     """
#     Brute Force matching
#     :param img1: image 1
#     :param img2: image 2
#     :param kp1: keypoints of frame 1
#     :param kp2: keypoints of frame 2
#     :param des1: descriptor of frame 1
#     :param des2: descriptor of frame 2
#     :return:
#     """
#     if (flag == "SIFT" or flag == "sift"):
#         # SIFT
#         bf = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=False)
#     else:
#         # ORB
#         bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=False)
#     ms = bf.match(des1, des2)
#     ms = sorted(ms, key=lambda x: x.distance)
#     img3 = cv2.drawMatches(img1, kp1, img2, kp2, ms, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#     # cv2.imshow("Matches", img3)
#     # cv2.waitKey(0)
#     return ms

def FLANN(img1, img2, kp1, kp2, des1, des2, flag):
    """
        2. FLANN matching
        :param img1: image 1
        :param img2: image 2
        :param kp1: keypoints of frame 1
        :param kp2: keypoints of frame 2
        :param des1: descriptor of frame 1
        :param des2: descriptor of frame 2
        :return:
        """
    print("img1",len(kp1),len(kp2),len(des1),len(des2))    
    if (flag == "SIFT" or flag == "SURF"):
        # SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE,
                            trees=5)
        search_params = dict(check=100)
    else:
        # ORB
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        
        
        search_params = dict(check=100)
    # define FLANN parameter
    # print('des1',des1.shape)
    # print('des2',des2.shape)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    return matches


"""
optimization matching results
RANSAC(RANdom SAmple Consensus)
"""


def RANSAC(img1, img2, kp1, kp2, matches,MIN_MATCH_COUNT):
    # MIN_MATCH_COUNT = 500
    # Good matches need to be stored according to lowe's ratio test.
    matchType = type(matches[0])
    good = []
    bad=[]
    print(len(matches))
    print(matchType)
    if isinstance(matches[0], cv2.DMatch):
        # Search for matching
        good = matches
    else:
        # knnMatch
        ratio_thresh = 0.75
        for m, n in matches:
            # print(m.distance,n.distance)
            if m.distance < ratio_thresh * n.distance:
                good.append(m)
            # else:
            #     bad.append(m)
            # if (((kp1[m.queryIdx].pt)[0]-(kp2[m.trainIdx].pt)[0])/((kp1[m.queryIdx].pt)[1]-(kp2[m.trainIdx].pt)[1]))>0.0877589:
            #    bad.append(m) 
            #     #gradient & length, x +,y -
            #     #if gradient 
                 #2 points from image 1 and 2, gradient < 0.0872665 radians
                
    img = cv2.drawMatches(img1, kp1, img2, kp2, good, None,  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("ratio", img)
    cv2.imwrite("after ration.jpg",img)
    print('number',len(good))
    
    if len(good) > MIN_MATCH_COUNT:
        #One list contains the coordinates of key points in the query image, and the other list contains the coordinates of matching key points in the scene:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        print('pt1',kp1[0].pt,kp1[0].size,kp1[0].angle)
        print('pt1',kp2[0].pt,kp2[0].size,kp2[0].angle)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # M: 3x3 Homography matrix. last number is the error of the transformation between points on original images and target images.
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # M, mask = cv2.findFundamentalMat(src_pts, dst_pts, method=cv2.RANSAC,ransacReprojThreshold=0.9, confidence=0.99)
        print('M',M)
        matchesMask = mask.ravel().tolist()

        # h, w = img1.shape
        # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # dst = cv2.perspectiveTransform(pts, M)
        #
        # img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print
        "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    # print('draw paras',len(draw_params))
    outliermask=[]
    for i in range(len(matchesMask)):
        if matchesMask[i]==0:
            outliermask.append(1)
        if matchesMask[i]==1:
            outliermask.append(0)
            
    # print('outlierMask',outliermask)        
    # print('matchesMask',matchesMask)
    # good_match_len=len(good)
    good_match_inliers=matchesMask.count(1)
    print('inliers',good_match_inliers)
    # print()
    # bad_match_len=outliermask.count(1)
    # match_outliers=matchesMask.count(0)
    # print(good_match_len,bad_match_len)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    draw_params1 = dict(matchColor=(0, 0, 255),  # draw matches in green color
                        singlePointColor=None,
                        matchesMask=outliermask,  # draw inliers without mask
                        flags=2)
    # print('matched',kp1.size,kp2.size)
    # img33 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params1)
    # print("good len",len(good))
    # print("bad len",len(bad))
    # print("kp1",len(kp1))
    img33 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params1) #outlier drawing
    cv2.imshow("before", img33)
    cv2.imwrite('matching FLANN00_00000v2sift outliers.jpg',img33)
    cv2.imshow("now", img3)
    cv2.imwrite('matching FLANN00_00000v2sift.jpg',img3)
    cv2.waitKey(20)
    # return good_match_len

# img1=Image.open(os.path.join(KITTI360Path,'data_2d_raw/2013_05_28_drive_0007_sync/image_00/data_rect/0000000003.png'))
# img1=Image.open(os.path.join(KITTI360Path,'data_2d_raw/2013_05_28_drive_0005_sync/image_00/data_rect/0000000001.png'))
img1=Image.open(os.path.join(KITTI360Path,'Test_data/real_world_images/00000.jpg'))
# img2=Image.open(os.path.join(KITTI360Path,'data_rendered_image/2013_05_28_drive_0005_sync','frame_02.jpg'))
# img2=Image.open(os.path.join(KITTI360Path,'data_rendered_image/2013_05_28_drive_0005_sync','frame_00.png'))
# img2=Image.open(os.path.join(KITTI360Path,'data_rendered_image/2013_05_28_drive_0005_sync','05_frame_00.png'))
img2=Image.open(os.path.join(KITTI360Path,'Test_data/rendered_images/00000.png'))
x=img1.width
y=img1.height

img1 = cv2.imread(os.path.join(KITTI360Path,'Test_data/real_world_images/00000.jpg'))
# img1 = cv2.imread(os.path.join(KITTI360Path,'data_2d_raw/2013_05_28_drive_0005_sync/image_00/data_rect/0000000001.png'))
# img2 = cv2.imread(os.path.join(KITTI360Path,'frame.jpg'))

# img2 = cv2.imread(os.path.join(KITTI360Path,'data_rendered_image/2013_05_28_drive_0005_sync','05_frame_00.png'))
# img2 = cv2.imread(os.path.join(KITTI360Path,'data_rendered_image/2013_05_28_drive_0005_sync','capture1_1.jpg'))
img2 = cv2.imread(os.path.join(KITTI360Path,'Test_data/rendered_images/00000.png'))
# img1=cv2.resize(img1,(x,y))
img2=cv2.resize(img2,(x,y))

# kp1, des1 = ORB(img1)
# kp2, des2 = ORB(img2)


# kp1, des1 = SURF(img1)
# kp2, des2 = SURF(img2)


kp1, des1 = SIFT(img1)
kp2, des2 = SIFT(img2)


# matches = Brute(img1, img2, kp1, kp2, des1, des2, "SURF")
# matches = FLANN(img1, img2, kp1, kp2, des1, des2, "SURF")
matches = FLANN(img1, img2, kp1, kp2, des1, des2, "SIFT")
# matches = Brute(img1, img2, kp1, kp2, des1, des2, "SIFT")
# matches = Brute(img1, img2, kp1, kp2, des1, des2, "ORB")
# matches = FLANN(img1, img2, kp1, kp2, des1, des2, "ORB")
RANSAC(img1, img2, kp1, kp2, matches,MIN_MATCH_COUNT=10)

# frame_idx=[]
# frame_idx_orb=[]
# matching_num_orb=[]
# frame_idx_sift=[]
# matching_num_sift=[]
# frame_idx_surf=[]
# matching_num_surf=[]
# # realimages_path=os.path.join(KITTI360Path,'Test_data/real_world_images')
# def imagepath(path):
#     images_path=os.path.join(KITTI360Path,'Test_data',path)
# # print(pc_path)
#     realfilelist = os.listdir(images_path)
#     realfilelist =sorted(realfilelist)
#     return realfilelist





# print(real_world_images)

# realimages_path=os.path.join(KITTI360Path,'Test_data/real_world_images')
# # print(pc_path)
# realfilelist = os.listdir(realimages_path)
# realfilelist =sorted(realfilelist)

# realimages_path=os.path.join(KITTI360Path,'Test_data/real_world_images')
# # print(pc_path)
# realfilelist = os.listdir(realimages_path)
# realfilelist =sorted(realfilelist )

# realimages_path=os.path.join(KITTI360Path,'Test_data/real_world_images')
# # print(pc_path)
# realfilelist = os.listdir(realimages_path)
# realfilelist =sorted(realfilelist)

# # print('file',filelist)
# print(len(realfilelist))


# def featurematchmetric(ByMatch,MIN_MATCH_COUNT):
#     real_world_images=imagepath('real_world_images')
#     rendered_images=imagepath('rendered_images')
#     assert(len(real_world_images)==len(rendered_images)) 
#     for i in range(len(real_world_images)):
#         img1=Image.open(os.path.join(KITTI360Path,'Test_data/real_world_images',real_world_images[i]))
#         img2=Image.open(os.path.join(KITTI360Path,'Test_data/rendered_images',rendered_images[i]))
#         x=img1.width
#         y=img1.height
#         img1 = cv2.imread(os.path.join(KITTI360Path,'Test_data/real_world_images',real_world_images[i]))
#         img2 = cv2.imread(os.path.join(KITTI360Path,'Test_data/rendered_images',rendered_images[i]))
#         img2=cv2.resize(img2,(x,y))
#         surf_kp1, surf_des1 = SURF(img1)
#         surf_kp2, surf_des2 = SURF(img2)
#         surf_matches=ByMatch(img1, img2, surf_kp1, surf_kp2, surf_des1, surf_des2, flag="SURF")
#         surf_matchnum=RANSAC(img1, img2, surf_kp1, surf_kp2, surf_matches,MIN_MATCH_COUNT)
#         orb_kp1, orb_des1 = ORB(img1)
#         orb_kp2, orb_des2 = ORB(img2) 
#         orb_matches=ByMatch(img1, img2, orb_kp1, orb_kp2, orb_des1, orb_des2, flag="ORB")
#         orb_matchnum=RANSAC(img1, img2, orb_kp1, orb_kp2, orb_matches,MIN_MATCH_COUNT)
#         sift_kp1, sift_des1 = SIFT(img1)
#         sift_kp2, sift_des2 = SIFT(img2) 
#         sift_matches=ByMatch(img1, img2, sift_kp1, sift_kp2, sift_des1, sift_des2, flag="SIFT")
#         sift_matchnum=RANSAC(img1, img2, sift_kp1, sift_kp2, sift_matches,MIN_MATCH_COUNT)       
        
        
#         frame_idx.append(i)
#         print('frame',frame_idx)
#         matching_num_surf.append(surf_matchnum)
#         print('surf',surf_matchnum)
#         matching_num_orb.append(orb_matchnum)
#         print('orb',orb_matchnum)
#         matching_num_sift.append(sift_matchnum)
#     plt.plot(frame_idx,matching_num_surf,label='surf matching')  # 画线并添加图例legend
#     plt.plot(frame_idx,matching_num_orb,label='orb matching')   # 画线并添加图例legend
#     plt.plot(frame_idx,matching_num_sift,label='sift matching') 
#     plt.legend()  # 展示图例legend
#     plt.xlabel('frame ID')  # 给 x 轴添加坐标轴信息
#     plt.ylabel('number of matching')  # 给 y 轴添加坐标轴信息
#     plt.title('descriptor of feature matching')  # 添加图片标题
   
#     return matching_num_surf,matching_num_orb,matching_num_sift     
# matching_num_surf,matching_num_orb,matching_num_sift=featurematchmetric(Brute,MIN_MATCH_COUNT=500)    

# print(frame_idx)    
#     img2=Image.open(os.path.join(KITTI360Path,'Test_data/rendered_images/00056.png'))
# x=img1.width
# y=img1.height

# img1 = cv2.imread(os.path.join(KITTI360Path,'Test_data/real_world_images/00056.jpg'))

# img2 = cv2.imread(os.path.join(KITTI360Path,'Test_data/rendered_images/00056.png'))

# img2=cv2.resize(img2,(x,y))
