#Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."
    print("RUN TIME: 10 seconds")
    print("Output: task1.png")
    sift = cv2.xfeatures2d.SIFT_create(400)
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)
    count = 0


    idx_desc1 = []
    idx_desc2 = []

    for i in range(len(descriptors_1)):
        for k in range(len(descriptors_2)):

            dis = np.sqrt(sum(np.square(descriptors_1[i] - descriptors_2[k])))

            if dis < 100:
                count = count + 1
                #print(i, k)
                idx_desc1.append(i)
                idx_desc2.append(k)

    kp11 = []
    kp22 = []

    kp1_list = []
    kp2_list = []
    for i in idx_desc1:
       # print(keypoints_1[i])
        kp1_list.append(keypoints_1[i].pt)
        kp11.append(keypoints_1[i])
    for j in idx_desc2:
       # print(keypoints_2[j])
        kp2_list.append(keypoints_2[j].pt)
        kp22.append(keypoints_2[j])

    # img = cv2.drawKeypoints(img1, kp11, None)
    # cv2.imshow('image',img)
    # cv2.waitKey(0)
    # img = cv2.drawKeypoints(img2, kp22, None)
    # cv2.imshow('image',img)
    # cv2.waitKey(0)

    src_pts = np.float32(kp1_list).reshape(-1, 1, 2)
    dst_pts = np.float32(kp2_list).reshape(-1, 1, 2)

    h, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    dst = cv2.warpPerspective(img1, h, ((img1.shape[1] + img2.shape[1]), img1.shape[0] + img2.shape[0]))

    for i in range(0, img2.shape[0]):
        for j in range(0, img2.shape[1]):

            # for i in range(0,dst.shape[0]):
            #   for j in range(0,dst.shape[1]):

            # dst[0:img2.shape[0], 0:img2.shape[1]] = img2
            if np.sum(dst[i][j]) > 0:
                # dst[i][j][0]=0
                # dst[i][j][1]=0
                # dst[i][j][2]=0
                # dst[i][j]=dst[dst.shape[0]-10][dst.shape[1]-10]
                # dst[i][j]=(dst[i][j]+img2[i][j])/2
                if np.sum(dst[i][j]) > np.sum(img2[i][j]):
                    dst[i][j] = dst[i][j]
                else:
                    dst[i][j] = img2[i][j]


            else:
                dst[i][j] = img2[i][j]
    # dst[0:img1.shape[0], 0:img1.shape[1]] = img1

    cv2.imshow("image",dst)
    cv2.waitKey(0)
    cv2.imwrite(savepath,dst)

    return
if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)

