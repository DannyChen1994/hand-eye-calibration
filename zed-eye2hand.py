"""
    指定文件夹下图片的棋盘格检测，并求出棋盘格相对于相机的姿态 x y z rx ry rz 轴角形式
    """
import numpy as np
import time
import cv2
import os
import math
from math import degrees as dg
from tqdm import tqdm
#import cv2.aruco as aruco


path="E:/WorkSpace/programs/ZED-Open3d/Eye2Hand/Data/image/"
# 相机内参 已标定 此处为ZED2 左目相机内存
camera_matrix = np.array([  [1058.6899, 0,  951.83],
                            [0,   1058.0100,550.4310],
                            [0,   0,    1]])
# print("Camera Matrix :\n {0}".format(camera_matrix))
dist_coeffs = np.array([ [-0.0404,0.0104,0.0001,0.0003,-0.0053] ])  # 相机畸变系数k1 k2 p1 p2 p3

# 查找棋盘格 角点
# 阈值
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 棋盘格参数
corners_vertical = 8    # 纵向角点个数;
corners_horizontal = 11  # 水平角点个数;
chessboard_size = (corners_vertical, corners_horizontal)
side_length=25  #棋盘格边长
# 世界坐标系下的物体位置矩阵（Z=0）
objp = np.zeros((np.prod(chessboard_size),3),dtype=np.float32)
# 通过np.mgrid生成对象的xy坐标点，每个棋盘格大小是13mm 不同的坐标轴设定会得到不同的坐标
# 最终得到z=0的objp为(0,0,0), (1*13,0,0), (2*13,0,0) ,...
#objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)*13
objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)*side_length
#objp[:,:2]=-objp[:,:2]
# 初始化三维坐标系
axis = np.float32([[200, 0, 0], [0, 200, 0], [0, 0, -200]]).reshape(-1, 3)  # 坐标轴

def draw(img, corners, imgpts):#参考，坐标系原点正确，坐标轴方向不一定是对应的
    """
    在图片上画出三维坐标轴
    :param img: 图片原数据
    :param corners: 图像平面点坐标点
    :param imgpts: 三维点投影到二维图像平面上的坐标
    :return:
    """

    # corners[0]是图像坐标系的坐标原点；imgpts[0]-imgpts[3] 即3D世界的坐标系点投影在2D世界上的坐标
    corner = tuple(corners[0].ravel())
    # 沿着3个方向分别画3条线
    ConersPixelXY = np.around(corner)
    ConersPixelXYInt = ConersPixelXY.astype(int) #转int
    P1=np.around(tuple(imgpts[0].ravel())).astype(int)
    P2=np.around(tuple(imgpts[1].ravel())).astype(int)
    P3=np.around(tuple(imgpts[2].ravel())).astype(int)
    cv2.line(img, ConersPixelXYInt, P1, (255, 0, 0), 5)
    cv2.line(img, ConersPixelXYInt, P2, (0, 255, 0), 5)
    cv2.line(img, ConersPixelXYInt, P3, (0, 0, 255), 5)
    return img

def find_corners_sb(img,newdir):
    """
    查找棋盘格角点函数 SB升级款
    :param img: 处理原图
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点;
    ret, corners = cv2.findChessboardCornersSB(gray, chessboard_size, cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
    if ret:
        # 显示角点
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        # rvec: 旋转向量 tvec: 平移向量
        _, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)     # 解算位姿，objp次序需要和corners对应，否则求出旋转矩阵不一定匹配
        sita_x = dg(rvec[0][0])                                                     # 测试中默认从右上角开始，向下x+，向左y+
        sita_y = dg(rvec[1][0])
        sita_z = dg(rvec[2][0])

        print("rotation vector is  ", rvec)
        XYZRxRyRz=np.append(tvec,rvec)
        np.savetxt(newdir,XYZRxRyRz,fmt='%0.5f')    #保存x y z rx ry rz 轴角形式，和ur机械臂一致

        distance = math.sqrt(tvec[0]**2+tvec[1]**2+tvec[2]**2)  # 计算距离
        rvec_matrix = cv2.Rodrigues(rvec)[0]    # 旋转向量->旋转矩阵
        proj_matrix = np.hstack((rvec_matrix, tvec))    # hstack: 水平合并
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]  # 欧拉角
        pitch, yaw, roll = eulerAngles[0], eulerAngles[1], eulerAngles[2]
        cv2.putText(img, "dist: %.2fcm, yaw: %.2f, pitch: %.2f, roll: %.2f" % (distance, yaw, pitch, roll), (10, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        #cv2.imshow('frame', frame)
        # 计算三维点投影到二维图像平面上的坐标
        imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
        # 把坐标显示图片上
        print("Rotation_matrix is  ", proj_matrix)
        
        img = draw(img, corners, imgpts)

    else:
        cv2.putText(img, "Unable to Detect Chessboard", (20, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3) 
        #cv2.imshow('frame', frame)

    cornersXY=np.squeeze(corners)
    #ConersPixelXY = np.around(corners)#像素点四舍五入，还是float
    #ConersPixelXYInt = ConersPixelXY.astype(int) #转int
    #print(cornersXY)    #角点像素坐标的二维数组，从右上开始往下排列
    return cornersXY

def find_corners(img):
    """
    查找棋盘格角点函数
    :param img: 处理原图
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点;
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                             cv2.CALIB_CB_FAST_CHECK +
                                             cv2.CALIB_CB_FILTER_QUADS)
    if ret:
        # 精细查找角点
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # 显示角点
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)

def GenerateMatCam2hand():
    #cap = cv2.VideoCapture(0)
    #font = cv2.FONT_HERSHEY_SIMPLEX # 字体
    #num = 0
    # 1.创建显示窗口
    cv2.namedWindow("img", 0)
    cv2.resizeWindow("img", 960, 540)
     # 2.循环读取标定图片
    filelist=os.listdir(path)
    for files in filelist:
    #for i in range(0, 1):
        olddir=os.path.join(path,files)
        #file_path = ('./image/left%02d.jpg' % i)
        #file_path = ('./image/left%02d.jpg' % i)
        img_src = cv2.imread(olddir)
        newdir = olddir[:-4]+".txt"
        if img_src is not None:
            # 执行查找角点算法
            #img_src = cv2.cvtColor(img_src, 1)    #4通道转三通道
            ConersXY=find_corners_sb(img_src,newdir)   #返回二维数组，x为图像向右像素，y向下
            

            # 显示图片
           
            cv2.imshow("img", img_src)
            cv2.waitKey(0)



if __name__ == '__main__':
    GenerateMatCam2hand()