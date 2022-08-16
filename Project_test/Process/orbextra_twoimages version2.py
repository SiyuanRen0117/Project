import numpy as np
import cv2
import os
from PIL import Image

#Main path

KITTI360Path="/media/ren/EXTERNAL_USB/KITTI360_DATASET/"
def ORB_Feature(img1, img2):

    # 初始化ORB
    orb = cv2.ORB_create(2000)

    # 寻找关键点
    kp1 = orb.detect(img1)
    kp2 = orb.detect(img2)

    # 计算描述符
    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)

    # 画出关键点
    outimg1 = cv2.drawKeypoints(img1, keypoints=kp1, outImage=None)
    outimg2 = cv2.drawKeypoints(img2, keypoints=kp2, outImage=None)
    cv2.imshow('outimg1',outimg1)
    cv2.imshow('outimg2',outimg2)
    # outimg3 = np.hstack([outimg1, outimg2])
    # cv2.imshow("Key Points", outimg3)
    # cv2.waitKey(0)
    # 初始化 BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # 对描述子进行匹配
    match = bf.match(des1, des2)
    # 计算最大距离和最小距离
    min_distance = match[0].distance
    max_distance = match[0].distance
    for x in match:
        if x.distance < min_distance:
            min_distance = x.distance
        if x.distance > max_distance:
            max_distance = x.distance


    # 筛选匹配点

    good_match = []
    for x in match:
        if x.distance <= max(2 * min_distance, 30):
            good_match.append(x)

    # 绘制匹配结果
    draw_match(img1, img2, kp1, kp2, good_match)

def draw_match(img1, img2, kp1, kp2, match):
    outimage = cv2.drawMatches(img1, kp1, img2, kp2, match, outImg=None)
    cv2.imshow("Match Result", outimage)
    
    RANSAC(img1, img2, kp1, kp2, match)
    cv2.waitKey(0)

def RANSAC(img1, img2, kp1, kp2, match):
    MIN_MATCH_COUNT = 200
    # store all the good match as per Lowe's ratio test.
    matchType = type(match[0])
    good = []
    print(matchType)
    if isinstance(match[0], cv2.DMatch):
        # Search for matching
        good = match
    else:
        # knnMatch
        for m, n in match:
            if m.distance < 0.6 * n.distance:
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
        "Not enough match are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
        matchesMask = None
    draw_params = dict(matchColor=(0, 255, 0),  # draw match in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    draw_params1 = dict(matchColor=(0, 255, 0),  # draw match in green color
                        singlePointColor=None,
                        matchesMask=None,  # draw only inliers
                        flags=2)

    img33 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params1)

    cv2.imshow("before", img33)
    cv2.imshow("now", img3)
    cv2.imwrite('matching testdata_orb.jpg',img3)
    cv2.waitKey(20)
    
if __name__ == '__main__':
 
    
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
    # image1 = cv2.imread('1.png')
    # image2 = cv2.imread('2.png')
    ORB_Feature(img1, img2)
