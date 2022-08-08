import numpy as np
import cv2
import os
from PIL import Image
from matplotlib import pyplot as plt

#Main path

KITTI360Path="/media/ren/EXTERNAL_USB/KITTI360_DATASET/"

print(KITTI360Path)



def ORB(img):
    """
     ORB detector
    """
    orb = cv2.ORB_create(nfeatures=2000)
    """find keypoint, and calculate descriptor"""
    kp, des = orb.detectAndCompute(img, None)

    # plot keypoints
    img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)

 
    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(img2[:, :, ::-1])
    plt.xticks([]), plt.yticks([])
    plt.show()
    return kp, des





"""
Feature matching method
(1). Brute Force matching
(2). FLAnn Matching

"""


def ByBFMatcher(img1, img2, kp1, kp2, des1, des2, flag="ORB"):
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
    # if (flag == "SIFT" or flag == "sift"):
    #     # SIFT
    #     bf = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=False)
    # else:
        # ORB
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=False)
    ms = bf.match(des1, des2 )
    ms = sorted(ms, key=lambda x: x.distance)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, ms, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Matches", img3)
    cv2.waitKey(0)
    return ms


def ByFlann(img1, img2, kp1, kp2, des1, des2, flag="ORB"):
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
    # if (flag == "SIFT" or flag == "sift"):
    #     # SIFT
    #     FLANN_INDEX_KDTREE = 1
    #     index_params = dict(algorithm=FLANN_INDEX_KDTREE,
    #                         trees=5)
    #     search_params = dict(check=50)
    # else:
    # ORB
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict(check=50)
    # parameters of FLANN
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    return matches


"""
optimization of matching results, applying RANSAC
"""

def RANSAC(img1, img2, kp1, kp2, matches):
    MIN_MATCH_COUNT = 100
    # store all the good matches as per Lower's ratio test.
    matchType = type(matches[0])
    good = []
    print(matchType)
    # print(matches)
    # if isinstance(matches[0], cv2.DMatch):
    #     # search for matching 
    #     good = matches
    # else:
    #     # search for knnMatch
    # for m, n in matches:
    #     if m.distance < 0.7 * n.distance:
    #         good.append(m)
    
    for m in matches:
    # print('m before',m.distance)
        if m.distance < max(2 * matches[0].distance, 30): #distance choose
        
        # if m.distance < 0.7 * n.distance:
        # print('m',m.distance)
            good.append(m)
    # good = matches
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)#feature descriptor index
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)#trained descriptor index

        # M: 3x3 Homography matrix. last number is the error of the transformation between points on original images and target images
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 20.0)
        matchesMask = mask.ravel().tolist()#make mask as a 1-dimension array

        # h, w = img1.shape
        # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # dst = cv2.perspectiveTransform(pts, M)
        
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

    img33 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params1)

    cv2.imshow("before", img33)
    cv2.imshow("now", img3)
    cv2.imwrite('matching results.jpg',img3)
    cv2.waitKey(0)


# img1=Image.open(os.path.join(KITTI360Path,'data_2d_raw/2013_05_28_drive_0007_sync/image_00/data_rect/0000000003.png'))
img1=Image.open(os.path.join(KITTI360Path,'data_2d_raw/2013_05_28_drive_0005_sync/image_00/data_rect/0000000003.png'))
# img1=Image.open(os.path.join(KITTI360Path,'data_rendered_image/2013_05_28_drive_0007_sync','frame_02.jpg'))
img2=Image.open(os.path.join(KITTI360Path,'data_rendered_image/2013_05_28_drive_0005_sync','frame_00.png'))
# img2=Image.open(os.path.join(KITTI360Path,'data_rendered_image/2013_05_28_drive_0007_sync','capture2.jpg'))
# img2=Image.open(os.path.join(KITTI360Path,'data_rendered_image/2013_05_28_drive_0007_sync','capture2_2.jpg'))
x=img1.width
y=img1.height

# img1 = cv2.imread(os.path.join(KITTI360Path,'data_2d_raw/2013_05_28_drive_0007_sync/image_00/data_rect/0000000003.png'))
img1 = cv2.imread(os.path.join(KITTI360Path,'data_2d_raw/2013_05_28_drive_0005_sync/image_00/data_rect/0000000003.png'))
# img2 = cv2.imread(os.path.join(KITTI360Path,'frame.jpg'))

# img2 = cv2.imread(os.path.join(KITTI360Path,'data_rendered_image/2013_05_28_drive_0005_sync','05_frame_00.png'))
# img2 = cv2.imread(os.path.join(KITTI360Path,'data_rendered_image/2013_05_28_drive_0005_sync','capture2.jpg'))
img2 = cv2.imread(os.path.join(KITTI360Path,'data_rendered_image/2013_05_28_drive_0005_sync','frame_00.png'))
# img1=cv2.resize(img1,(x,y))
img2=cv2.resize(img2,(x,y))

kp1, des1 = ORB(img1)
kp2, des2 = ORB(img2)
matches = ByBFMatcher(img1, img2, kp1, kp2, des1, des2, "ORB")
# matches = ByFlann(img1, img2, kp1, kp2, des1, des2, "ORB")
print('matching 05frame00',matches)
RANSAC(img1, img2, kp1, kp2, matches)
