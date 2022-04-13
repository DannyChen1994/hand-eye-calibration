import cv2
import numpy as np
import transforms3d as tfs
import math

path1="E:/WorkSpace/programs/ZED-Open3d/Eye2Hand/image/"   #标定板相对于相机pose
path2="E:/WorkSpace/programs/openCVTest/Data/"      #机械臂pose
numData=3
chess_to_cam = np.zeros(shape=(numData,6))
end_to_base = np.zeros(shape=(numData,6)) 
for i in  range(0, numData, 1):             #读取pose数据
    txtDir=path1+"poses"+str(i)+".txt"   #shape 可能是shape(6,1)
    txt=np.loadtxt(txtDir).reshape(1,6) #读取txt
    chess_to_cam[i,:]= txt              #np.append(chess_to_cam, txt,axis=0)
    
    RobotPose=path2+"poses"+str(i)+".txt"   #shape 可能是shape(6,1)
    txt2=np.loadtxt(RobotPose).reshape(1,6) #读取txt
    end_to_base[i,:]= txt2              


# # 标定板在相机坐标系下的平移 旋转向量
# chess_to_cam = [[ 0.05418929, -0.06419668,  0.39711058, -2.9976273, -0.67330669, -0.28534635],
#                 [-0.11631906, -0.07964155,  0.39275717, 3.21465599, 0.03229772, 0.22894979],
#                 [0.1221848,  0.05505763, 0.35671759, 2.93567566, 1.03199128, 0.22884697],
#                 [-0.07425948,  0.03702354,  0.33609526, 3.06939635, 0.24315356, 0.41800747],
#                 [-0.00932829,  0.10122347,  0.35569078, 3.08788962, 0.10030853, 0.01667269],
#                 [-0.04010734,  0.03727042,  0.41037381, -3.17842059, -0.1270912, -0.158062],
#                 [0.03476735, -0.09092298,  0.44136333, -3.00925257, -0.30915832, -0.21673868],
#                 [0.02215647, 0.00334369, 0.42765002, -3.04357826, -0.46354661, -0.06893606],
#                 [0.16825633, 0.00080361, 0.40465769, 3.0987636,  0.69388335, 0.04170716],
#                 [-0.1019604,  -0.00548816,  0.43585263, 3.12137797, -0.1709356, 0.01390725],
#                 [-0.11973229,  0.03800298,  0.39819154, 3.03525967, -0.6982821, -0.09134344],
#                 [0.09775245, 0.00264186, 0.36520473, -3.0113346, -0.04030129, 0.35429879]]

# # 机械臂末端在基坐标系下的姿态（x y z rx ry rz）
# end_to_base = [[-0.352837, -0.478216, 0.031524, -90.248, -22.131, 51.770],
#                [-0.380905, -0.512619, 0.059280, -89.829, 3.205, 51.563],
#                [-0.347814, -0.398185, -0.037019, -93.345, -35.0, 54.113],
#                [-0.407478, -0.470825, -0.076273, -89.082, -5.482, 53.205],
#                [-0.397457, -0.459013, -0.120213, -90.145, 0.269, 43.073],
#                [-0.362895, -0.501056, -0.059730, -90.100, -1.064, 46.929],
#                [-0.314285, -0.498408, 0.057705, -89.496, -9.043, 47.461],
#                [-0.348414, -0.508882, -0.042078, -88.810, -13.996, 45.284],
#                [-0.285571, -0.425445, -0.004844, -87.643, -22.045, 44.916],
#                [-0.367625, -0.534044, 0.014010, -91.886, 9.570, 45.284],
#                [-0.390861, -0.498887, 0.002275, -92.364, 29.380, 39.0],
#                [-0.356676, -0.465285, -0.058769, -85.400, 2.048, 30.800]]
chess_to_cam_R,chess_to_cam_T = [],[]
end_to_base_R,end_to_base_T = [],[]
for chess_cam in chess_to_cam:
    cc=chess_cam[3:6]
    cc_R, j = cv2.Rodrigues((cc[0],cc[1],cc[2]))
    chess_to_cam_R.append(cc_R)
    chess_to_cam_T.append(np.array(chess_cam[0:3]).reshape(3,1))
for end_base in end_to_base:
    ed=end_base[3:6]
    ed_R, j2 = cv2.Rodrigues((ed[0],ed[1],ed[2]))
    #end_to_base_R.append(tfs.euler.euler2mat(math.radians(ed[0]),math.radians(ed[1]),math.radians(ed[1]),axes='sxyz'))  #注意欧拉角顺序
    end_to_base_R.append(ed_R)
    end_to_base_T.append(np.array(end_base[0:3]).reshape(3,1))
 
print(chess_to_cam_R)
print(chess_to_cam_T)
print(end_to_base_R)
print(end_to_base_T)
cam_to_end_R,cam_to_end_T = cv2.calibrateHandEye(end_to_base_R,end_to_base_T,chess_to_cam_R,chess_to_cam_T,
                                                 method=cv2.CALIB_HAND_EYE_TSAI)
print(cam_to_end_R)
print(cam_to_end_T)
cam_to_end_RT = tfs.affines.compose(np.squeeze(cam_to_end_T), cam_to_end_R, [1, 1, 1])
print("标定结果：\n",cam_to_end_RT)
 
# 结果验证，原则上来说，每次结果相差较小
for i in range(0,numData):
    RT_end_to_base=np.column_stack((end_to_base_R[i],end_to_base_T[i].reshape(3,1)))
    RT_end_to_base=np.row_stack((RT_end_to_base,np.array([0,0,0,1])))
    # print(RT_end_to_base)
    RT_chess_to_cam=np.column_stack((chess_to_cam_R[i],chess_to_cam_T[i].reshape(3,1)))
    RT_chess_to_cam=np.row_stack((RT_chess_to_cam,np.array([0,0,0,1])))
    # print(RT_chess_to_cam)
    RT_chess_to_base=RT_end_to_base@cam_to_end_RT@RT_chess_to_cam #固定的棋盘格相对于机器人基坐标系位姿
    # RT_chess_to_base=np.linalg.inv(RT_chess_to_base)
    print('第',i,'次')
    print(RT_chess_to_base)
    print('')