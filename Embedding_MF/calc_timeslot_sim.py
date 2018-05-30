#计算train集合中，24小时的签到行为相似度
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

file_in = "../data_5months/train/g_train_time_sim_matrix.txt"
file_out = "../data_5months/train/g_smooth_train_time_sim_matrix.txt"

train_time_sim_matrix = np.loadtxt(open(file_in,"r",encoding="utf-8"),delimiter=",")
time_sim_dict = {}
matrix = np.zeros((24,24),dtype=float)
for i in range(1,len(train_time_sim_matrix[0,:])):
    print("======Calc {0} Column=======".format(i))
    time_i_feature = train_time_sim_matrix[1:,i]
    print(time_i_feature)
    time_i_2norm = linalg.norm(time_i_feature,2)
    print(time_i_2norm)
    time_sim_dict[i] = []
    for j in range(1,len(train_time_sim_matrix[0,:])):
        time_j_feature = train_time_sim_matrix[1:,j]
        time_j_2norm = linalg.norm(time_j_feature,2)
        numerator = float(np.dot(time_i_feature, time_j_feature))
        # print(numerator)
        denominator = time_i_2norm * time_j_2norm
        cos = numerator/denominator
        # similarity = 0.5 + 0.5*cos
        time_sim_dict[i].append(cos)
    print("Done!")
    matrix[i-1,:] = time_sim_dict[i]
    i += 1
# np.savetxt(file_out,matrix,delimiter=",",fmt="%f")
# 24维*24维的时间相似矩阵

plt.xlabel("Time")
plt.ylabel("Cosine Similarity")
x_array =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
# plt.subplot(221)
# for i in range(1,25):
#     plt.plot(x_array,time_sim_dict[i],marker="o")
for i in range(0,6):
    # print(time_sim_dict[i])
    plt.subplot(221)
    plt.plot(x_array,time_sim_dict[i+1],marker="o")
for i in range(6,12):
    # print(time_sim_dict[i])
    plt.subplot(222)
    plt.plot(x_array,time_sim_dict[i+1],marker="o")
for i in range(12,18):
    # print(time_sim_dict[i])
    plt.subplot(223)
    plt.plot(x_array,time_sim_dict[i+1],marker="o")
for i in range(18,24):
    # print(time_sim_dict[i])
    plt.subplot(224)
    plt.plot(x_array,time_sim_dict[i+1],marker="o")
plt.legend()
plt.draw()
plt.savefig("time_sim_pic_total2",dpi=600)
plt.show()
