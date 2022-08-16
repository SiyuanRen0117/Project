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
    orb = cv2.ORB_create(2000)
    """find keypoint, and calculate descriptor"""
    kp, des = orb.detectAndCompute(img, None)

    # plot keypoints
    img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)

 
    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(img2[:, :, ::-1])
    plt.xticks([]), plt.yticks([])
    plt.show()
    return kp, des



def SIFT(img):
    """
     SIFT detector
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # SIFT keypoint detector
    # sift instantiation 
    sift = cv2.xfeatures2d.SIFT_create()

    # sift detect and compute 
    kp, des = sift.detectAndCompute(gray, None)
    
    # kp: information of keypoint, including position, scale, direction
    # des: keypoint descriptor, corresponds to a feature vector of 128 gradient information

    # image show
    cv2.drawKeypoints(img, kp, img, (0, 255, 0))

    
    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(img[:, :, ::-1])
    plt.xticks([]), plt.yticks([])
    plt.show()
    return kp, des


def SURF(img):
    surf = cv2.xfeatures2d.SURF_create()
    kp, des = surf.detectAndCompute(img, None)
    cv2.drawKeypoints(img, kp, img, (0, 255, 0))

    
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
    :param kp1: keypoints of frame 1
    :param kp2: keypoints of frame 2
    :param des1: descriptor of frame 1
    :param des2: descriptor of frame 2
    :return:
    """
    # if type(des1)!=NoneType and type(des2)!=NoneType:
    if (flag == "SIFT" or flag == "SURF"):
        # SIFT
        bf = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=True)
        
    else:
        # ORB
        # bf = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=True)
        bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
    print('des1',des1.shape)
    print('des2',des2.shape)
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
    if (flag == "SIFT" or flag == "SURF"):
        # SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE,
                            trees=5)
        search_params = dict(check=50)
    else:
        # ORB
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        
        
        search_params = dict(check=50)
    # define FLANN parameter
    print('des1',des1.shape)
    print('des2',des2.shape)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    return matches


"""
optimization matching results
RANSAC(RANdom SAmple Consensus)
"""


def RANSAC(img1, img2, kp1, kp2, matches):
    MIN_MATCH_COUNT = 1000
    # store all the good matches as per Lowe's ratio test.
    matchType = type(matches[0])
    good = []
    print(matchType)
    if isinstance(matches[0], cv2.DMatch):
        # Search for matching
        good = matches
    else:
        # knnMatch
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # M: 3x3 Homography matrix. last number is the error of the transformation between points on original images and target images.
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
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

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    draw_params1 = dict(matchColor=(0, 255, 0),  # draw matches in green color
                        singlePointColor=None,
                        matchesMask=None,  # draw only inliers
                        flags=2)
    # print('matched',kp1.size,kp2.size)
    img33 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params1)

    cv2.imshow("before", img33)
    cv2.imshow("now", img3)
    cv2.imwrite('matching testdata93_SURF20000.jpg',img3)
    cv2.waitKey(20)


# img1=Image.open(os.path.join(KITTI360Path,'data_2d_raw/2013_05_28_drive_0007_sync/image_00/data_rect/0000000003.png'))
# img1=Image.open(os.path.join(KITTI360Path,'data_2d_raw/2013_05_28_drive_0005_sync/image_00/data_rect/0000000001.png'))
img1=Image.open(os.path.join(KITTI360Path,'Test_data/real_world_images/00093.jpg'))
# img2=Image.open(os.path.join(KITTI360Path,'data_rendered_image/2013_05_28_drive_0005_sync','frame_02.jpg'))
# img2=Image.open(os.path.join(KITTI360Path,'data_rendered_image/2013_05_28_drive_0005_sync','frame_00.png'))
# img2=Image.open(os.path.join(KITTI360Path,'data_rendered_image/2013_05_28_drive_0005_sync','05_frame_00.png'))
img2=Image.open(os.path.join(KITTI360Path,'Test_data/rendered_images/00094.png'))
x=img1.width
y=img1.height

img1 = cv2.imread(os.path.join(KITTI360Path,'Test_data/real_world_images/00093.jpg'))
# img1 = cv2.imread(os.path.join(KITTI360Path,'data_2d_raw/2013_05_28_drive_0005_sync/image_00/data_rect/0000000001.png'))
# img2 = cv2.imread(os.path.join(KITTI360Path,'frame.jpg'))

# img2 = cv2.imread(os.path.join(KITTI360Path,'data_rendered_image/2013_05_28_drive_0005_sync','05_frame_00.png'))
# img2 = cv2.imread(os.path.join(KITTI360Path,'data_rendered_image/2013_05_28_drive_0005_sync','capture1_1.jpg'))
img2 = cv2.imread(os.path.join(KITTI360Path,'Test_data/rendered_images/00094.png'))
# img1=cv2.resize(img1,(x,y))
img2=cv2.resize(img2,(x,y))


# kp1, des1 = ORB(img1)
# kp2, des2 = ORB(img2)


kp1, des1 = SURF(img1)
kp2, des2 = SURF(img2)


# kp1, des1 = SIFT(img1)
# kp2, des2 = SIFT(img2)


# matches = Brute(img1, img2, kp1, kp2, des1, des2, "SURF")
matches = FLANN(img1, img2, kp1, kp2, des1, des2, "SURF")
# matches = FLANN(img1, img2, kp1, kp2, des1, des2, "SIFT")
# matches = Brute(img1, img2, kp1, kp2, des1, des2, "SIFT")
# matches = Brute(img1, img2, kp1, kp2, des1, des2, "ORB")
# matches = FLANN(img1, img2, kp1, kp2, des1, des2, "ORB")
RANSAC(img1, img2, kp1, kp2, matches)