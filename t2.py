# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import json

def stitch(imgmark, N=4, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"
    sift = cv2.xfeatures2d.SIFT_create(500)
    print("STITCHING PROCESS GOING ON....")
    print("RUN TIME: 14 seconds (task2), 30 seconds(overlap array task1), 10 seconds (task3) + 8 seconds (Overlap array task3)")
    print("Output: task2.png, task3.png, t2_overlap.txt, t3_overlap.txt")
    #cv2.imshow("image", imgs[1])
    #cv2.waitKey(0)
    keypoints_1, descriptors_1 = sift.detectAndCompute(imgs[0], None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(imgs[1], None)
    keypoints_3, descriptors_3 = sift.detectAndCompute(imgs[2], None)
    keypoints_4, descriptors_4 = sift.detectAndCompute(imgs[3], None)


    count = 0
    import math

    idx_desc1 = []
    idx_desc2 = []

    for i in range(len(descriptors_1)):
        for k in range(len(descriptors_2)):

            dis = np.sqrt(sum(np.square(descriptors_1[i] - descriptors_2[k])))

            if dis < 60:


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

    # img = cv2.drawKeypoints(imgs[0], kp11, None)
    # cv2.imshow(img)
    # cv2.waitKey(0)
    # img = cv2.drawKeypoints(imgs[1], kp22, None)
    # cv2.imshow(img)
    # cv2.waitKey(0)
    src_pts = np.float32(kp1_list).reshape(-1, 1, 2)
    dst_pts = np.float32(kp2_list).reshape(-1, 1, 2)

    h, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    points0 = np.array([[0, 0], [0, imgs[0].shape[0]], [imgs[0].shape[1], imgs[0].shape[0]], [imgs[1].shape[1], 0]],
                       dtype=np.float32)
    points0 = points0.reshape((-1, 1, 2))
    points1 = np.array([[0, 0], [0, imgs[1].shape[0]], [imgs[1].shape[1], imgs[1].shape[0]], [imgs[1].shape[1], 0]],
                       dtype=np.float32)
    points1 = points1.reshape((-1, 1, 2))
    points2 = cv2.perspectiveTransform(points1, h)

    points = np.concatenate((points0, points2), axis=0)
    [x_min, y_min] = np.int32(points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(points.max(axis=0).ravel() + 0.5)
    h_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    h_final = h_translation.dot(h)

    result = cv2.warpPerspective(imgs[0], h_final, (x_max - x_min, y_max - y_min))
    result[abs(y_min):imgs[1].shape[0] + abs(y_min), abs(x_min): imgs[1].shape[1] + abs(x_min)] = imgs[1]

    keypoints_5, descriptors_5 = sift.detectAndCompute(result, None)
    idx_desc3 = []
    idx_desc5 = []

    for i in range(len(descriptors_5)):
        for k in range(len(descriptors_3)):

            dis = np.sqrt(sum(np.square(descriptors_5[i] - descriptors_3[k])))

            if dis < 100:
                count = count + 1
                #print(i, k)
                idx_desc5.append(i)
                idx_desc3.append(k)
    kp3_list = []
    kp5_list = []
    for i in idx_desc5:
        # print(keypoints_1[i])
        kp5_list.append(keypoints_5[i].pt)
        # kp11.append(keypoints_1[i])
    for j in idx_desc3:
        # print(keypoints_2[j])
        kp3_list.append(keypoints_3[j].pt)
        # kp22.append(keypoints_5[j])

    src_pts = np.float32(kp5_list).reshape(-1, 1, 2)
    dst_pts = np.float32(kp3_list).reshape(-1, 1, 2)

    h, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    points0 = np.array([[0, 0], [0, result.shape[0]], [result.shape[1], result.shape[0]], [imgs[2].shape[1], 0]],
                       dtype=np.float32)
    points0 = points0.reshape((-1, 1, 2))
    points1 = np.array([[0, 0], [0, imgs[2].shape[0]], [imgs[2].shape[1], result.shape[0]], [imgs[2].shape[1], 0]],
                       dtype=np.float32)
    points1 = points1.reshape((-1, 1, 2))
    points2 = cv2.perspectiveTransform(points1, h)

    points = np.concatenate((points0, points2), axis=0)
    [x_min, y_min] = np.int32(points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(points.max(axis=0).ravel() + 0.5)
    h_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    h_final = h_translation.dot(h)

    result1 = cv2.warpPerspective(result, h_final, (x_max - x_min, y_max - y_min))
    result1[abs(y_min):abs(imgs[2].shape[0]) + abs(y_min), abs(x_min): abs(imgs[2].shape[1]) + abs(x_min)] = imgs[2]

    keypoints_6, descriptors_6 = sift.detectAndCompute(result1, None)

    idx_desc4 = []
    idx_desc6 = []

    for i in range(len(descriptors_6)):
        for k in range(len(descriptors_4)):

            dis = np.sqrt(sum(np.square(descriptors_6[i] - descriptors_4[k])))

            if dis < 100:
                count = count + 1
                #print(i, k)
                idx_desc6.append(i)
                idx_desc4.append(k)

    kp4_list = []
    kp6_list = []
    for i in idx_desc6:
        # print(keypoints_1[i])
        kp6_list.append(keypoints_6[i].pt)
        # kp11.append(keypoints_1[i])
    for j in idx_desc4:
        # print(keypoints_2[j])
        kp4_list.append(keypoints_4[j].pt)
        # kp22.append(keypoints_5[j])
    src_pts = np.float32(kp6_list).reshape(-1, 1, 2)
    dst_pts = np.float32(kp4_list).reshape(-1, 1, 2)
    h, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    points0 = np.array([[0, 0], [0, result1.shape[0]], [result1.shape[1], result1.shape[0]], [imgs[3].shape[1], 0]],
                       dtype=np.float32)
    points0 = points0.reshape((-1, 1, 2))
    points1 = np.array([[0, 0], [0, imgs[3].shape[0]], [imgs[3].shape[1], result1.shape[0]], [imgs[3].shape[1], 0]],
                       dtype=np.float32)
    points1 = points1.reshape((-1, 1, 2))
    points2 = cv2.perspectiveTransform(points1, h)

    points = np.concatenate((points0, points2), axis=0)
    [x_min, y_min] = np.int32(points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(points.max(axis=0).ravel() + 0.5)
    h_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    h_final = h_translation.dot(h)

    result2 = cv2.warpPerspective(result1, h_final, (x_max - x_min, y_max - y_min))
    result2[abs(y_min):imgs[3].shape[0] + abs(y_min), abs(x_min): imgs[3].shape[1] + abs(x_min)] = imgs[3]

    cv2.imshow("image",result2)
    cv2.waitKey(0)
    cv2.imwrite(savepath,result2)

    m = []
    dict_desc = {1: descriptors_1, 2: descriptors_2, 3: descriptors_3, 4: descriptors_4}
    for ii in range(1, 5):
        d1 = dict_desc[ii]
       # print(ii)
        for jj in range(1, 5):
           # print(jj)
            d2 = dict_desc[jj]
            count = 0
            # m=0
            for i in range(len(d1) - 150):
                for k in range(len(d2) - 150):

                    dis = np.sqrt(sum(np.square(d1[i] - d2[k])))
                    if dis < 100:
                        count = count + 1
            if count > 2:
                m.append(1)
            else:
                m.append(0)
    overlap_arr = np.array(m)
    return overlap_arr
if __name__ == "__main__":
    #task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    #bonus
    overlap_arr2 = stitch('t3', savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
