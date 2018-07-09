# Foursquare全时间跨度的数据集，其特征矩阵构造

from math import radians, cos, sin, asin, sqrt
from multiprocessing import Process
import numpy as np
import collections

#从总geo列表得到将要构建的两个矩阵的维数
file_total_geo = "../f_data/valid_total/f_valid_total_poi_geo.txt"

#以下都是要生成的文件
file_poi_geo_sim = "../f_data/matrix/bugtest_geosim_threshold4.txt"

#矩阵的维度是所有poi的列表长度
def create_key(file_total_geo):
    '''
    :param poi_geo_dict: dict
    构建一个poi集合，作为共现矩阵的首行与首列
    poilist:是总数据量的poilist，并非只是训练集的poilist
    :return:
    '''
    poilist = []
    total_poi_geo_dict = {}
    with open(file_total_geo,"r",encoding="utf-8") as f:
        for line in f.readlines():
            poi_id = int(line.split(":")[0])
            poi_geo = []
            geo_str = line.split(":")[1]
            # print(geo_str)
            # print(geo_str.replace("[","").replace("]\n","").split(", "))
            poi_geo.append(float(geo_str.replace("[","").replace("]\n","").split(", ")[0]))       # strip("[]")
            poi_geo.append(float(geo_str.replace("[","").replace("]\n","").split(", ")[1]))
            # print(poi_geo)
            poilist.append(poi_id)
            print("加入新poi:{0} {1}".format(poi_id,poi_geo))
            if poi_id not in total_poi_geo_dict.keys():
                total_poi_geo_dict[poi_id] = poi_geo
        print("====poi总列表====")
        poilist = np.sort(poilist)
        print(poilist)
        f.close()
        return list(poilist),total_poi_geo_dict

# 为不同的线程选择不同的运行函数
def select_func_run(func_name, poilist, data, file):
    if func_name == "co_visit":
        co_visiting_matrix = create_covisiting_matrix(poilist, data)
        np.savetxt(file,co_visiting_matrix,delimiter=",",fmt="%d")
    elif func_name == "visit_time":
        visited_time_matrix = create_visitedtime_matrix(poilist, data)
        np.savetxt(file,visited_time_matrix,delimiter=",",fmt="%d")
    else:
        poi_geo_sim_matrix = create_poi_geosim_matrix(poilist, data)
        np.savetxt(file,poi_geo_sim_matrix,delimiter=",",fmt="%d")


# 构建poi的地理位置相似矩阵
def create_poi_geosim_matrix(poilist, data):
    '''
    :param poilist:
    建立矩阵，矩阵的长宽都是poi集合长度+1
    给矩阵每一元素赋值
    :return:
    '''
    print("初始化开始！")
    edge = len(poilist) + 1
    matrix = np.zeros((edge, edge),dtype=np.int32)

    matrix[0][1:] = np.array(poilist)
    i = 1
    j = 0
    while i <= edge and j < len(poilist):
        matrix[i][0] = poilist[j]
        i += 1
        j += 1
    print("geo首行首列赋值完成")

    for i in range(len(poilist)):
        pid1 = poilist[i]
        print("calc poi{0}".format(pid1))
        lat1 = data[pid1][0]
        lon1 = data[pid1][1]
        lat1,lon1 = map(radians, [lat1, lon1])
        for j in range(len(poilist)):
            pid2 = poilist[j]
            lat2 = data[pid2][0]
            lon2 = data[pid2][1]
            lat2,lon2 = map(radians, [lat2, lon2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * asin(sqrt(a))
            r = 6371
            distance = c * r
            # weight = 0.5+distance
            # matrix[i+1, j+1] = weight
            matrix[i+1, j+1] = distance
        # boundary = np.sort(matrix[i+1, 1:])[300]
        smaller = matrix[i+1, 1:] < 4
        bigger = matrix[i+1, 1:] >= 4
        matrix[i+1, 1:][smaller] = 1
        matrix[i+1, 1:][bigger] = 0
        print(matrix[i+1, 1:])

    return matrix

if __name__ == '__main__':
    # 获取所有合格的poilist
    poilist, total_poi_geo_dict = create_key(file_total_geo)
    # print(total_poi_geo_dict)

    p3 = Process(target=select_func_run, args=("poi_geo", poilist, total_poi_geo_dict, file_poi_geo_sim))
    # 进程3：所有poi地理相似矩阵构建

    p3.start()
    #

    p3.join()
