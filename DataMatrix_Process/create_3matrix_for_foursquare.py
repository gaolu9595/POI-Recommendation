# Foursquare全时间跨度的数据集，其特征矩阵构造

from math import radians, cos, sin, asin, sqrt
from multiprocessing import Process
import numpy as np
import collections

#从训练集中构建co_visit和visit_time矩阵,geo_sim矩阵不需要重新构建【因为与是否是训练集无关】
file = "../f_data/f_train_set.txt"
#从总geo列表得到将要构建的两个矩阵的维数
file_total_geo = "../f_data/valid_total/f_valid_total_poi_geo.txt"

#以下都是要生成的文件
file_total_user_poi_dict = "../f_data/valid_total/f_valid_total_user_visit.txt"
file_user_poi_dict = "../f_data/train/f_train_user_visit.txt"
file_time_poi_dict = "../f_data/train/f_train_visit_time.txt"
file_geo_poi_dict = "../f_data/train/f_train_poi_geo.txt"

file_co_visit = "../f_data/matrix/bugtest_covisit.txt"
# file_co_visit = "../f_data/matrix/f_train_co_visit_matrix.txt"
file_visit_time = "../f_data/matrix/f_train_time_sim_matrix.txt"
file_poi_geo_sim = "../f_data/matrix/f_valid_total_poi_sim_matrix.txt"

def readFile(file): #,poilist
    with open(file,"r",encoding="utf-8") as f:
        user_poi_dict = {}
        time_poi_dict = {}
        poi_geo_dict = {}
        for line in f.readlines():
            info = line.split("	")
            #id都可以是整型数据，经纬度是浮点数，时间现在先当做是字符串
            user_id = int(info[0][5:])
            time = info[3]
            poi_id = int(info[1][4:])
            geo_info = info[2].split(",")
            geo = [float(geo_info[0]),float(geo_info[1])]
            #构建训练集用户访问训练集poi列表的字典
            if user_id in user_poi_dict.keys():
                if poi_id not in user_poi_dict[user_id]:
                    user_poi_dict[user_id].append(poi_id)
            else:
                user_poi_dict[user_id] = [poi_id]
                print("训练集user：",user_id)
            #对时间格式进行预处理,获取签到行为的时间点(24小时制)
            time = int(time[0:2])
            #构建训练集poi被访问时间的字典，只选取部分时间内的poi签到记录
            if time in time_poi_dict.keys():
                #每个时间的poi列表是有重复元素的
                time_poi_dict[time].append(poi_id)
            else:
                time_poi_dict[time] = [poi_id]
            #构建训练集poi地理位置字典
            if poi_id not in poi_geo_dict.keys():
                poi_geo_dict[poi_id] = geo
                print("训练集poi：",poi_id)

        user_count = 0
        poi_count = 0
        print("====每个用户访问过的poi列表====")
        for user in user_poi_dict.keys():
            print(user,":",user_poi_dict[user])
            user_count+=1
        print("训练集中USER数目:",user_count)

        print("====每个时间点内poi被访问列表====")
        for time in time_poi_dict.keys():
            print(time,":",time_poi_dict[time])

        print("====每个poi的地理位置经纬度信息====")
        for poi in list(poi_geo_dict.keys()):
            print(poi,":",poi_geo_dict[poi])
            poi_count+=1
        print("训练集中POI数目:",poi_count)

        f.close()
        return user_poi_dict,time_poi_dict,poi_geo_dict

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

def format_data(user_poi_dict):
    # formatdata是每个用户访问的poi列表的二维数组
    formatdata = []
    for user in user_poi_dict.keys():
        value = user_poi_dict[user]
        formatdata.append(value)
    print("====每个用户访问的poi列表记录的集合====")
    print(formatdata)
    return formatdata

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

# 构建POI共现矩阵
def create_covisiting_matrix(poilist, data):
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
    print("co_visit首行首列赋值完成")

    # 把每个关键词在 format_data 中出现的行的序列 组成一个字典
    # 共现的判断就转化成了 比较两个关键词对应的list 交集个数的问题
    # train_poilist = list(poi_geo_dict.keys())
    appeardict = {}
    for poi in poilist:
        appearlist = []
        i = 0
        for each_line in data:
            if poi in each_line:
                appearlist.append(i)
            i += 1
        appeardict[poi] = appearlist
        # print("poi{0}出现行号列表:{1}".format(poi,appearlist))

    # 计算poi共现次数赋值给矩阵相应位置,考虑到其对称性
    for row in range(1, len(matrix)):
        if row != 1:
            for col in range(1, row):
                matrix[row][col] = matrix[col][row]
        for col in range(row, len(matrix)):
            # 跳过（0,0）位置的空元素
            print("covisit位置[{0},{1}]".format(row, col))
            if matrix[row][0] == matrix[0][col]:
                matrix[row][col] = len(set(appeardict[matrix[row][0]]))  # 对角线元素
            else:
                matrix[row][col] = len(set(appeardict[matrix[row][0]])&set(appeardict[matrix[0][col]]))
    return matrix

# 构建poi被访问时间相似矩阵
def create_visitedtime_matrix(poilist, data):
    '''
    :param poilist:
    :param data:
    建立矩阵，矩阵的高度是poi集合长度+1，宽度是24+1
    给矩阵每一元素赋值
    :return:
    '''
    print("初始化开始！")
    row_num = len(poilist) + 1
    col_num = 25
    matrix = np.zeros((row_num, col_num),dtype=np.int32)

    matrix[0][1:] = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
    i = 1
    j = 0
    while i <= row_num and j < len(poilist):
        matrix[i][0] = poilist[j]
        i += 1
        j += 1
    print("time首行首列赋值完成")
    # 读取time_poi_dict数据，给矩阵每个位置赋值
    for row in range(1, row_num):
        for col in range(1, col_num):
            print("time位置[{0},{1}]".format(row, col))
            # 使用标准库中的Collections，得到列表中每个元素出现的次数统计
            # Counter其实是dict的一个子类，是一个简单的计数器
            d = collections.Counter(data[col-1])
            if matrix[row][0] in d.keys():
                matrix[row][col] = d[matrix[row][0]]
            else:
                matrix[row][col] = 0
    return matrix

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

    # 计算poi之间距离，若距离大于m赋值为1，小于m赋值为0【根据经验，先取m为4km】
    for row in range(1, len(matrix)):
        if row != 1:
            for col in range(1, row):
                matrix[row][col] = matrix[col][row]
        for col in range(row, len(matrix)):
            # 跳过（0,0）位置的空元素
            print("geo位置[{0},{1}]".format(row, col))
            if matrix[row][0] == matrix[0][col]:
                matrix[row][col] = 0  # 对角线元素为0
            else:
                # 计算两个poi之间的经纬度距离
                lat1, lon1 = data[matrix[row][0]]
                lat2, lon2 = data[matrix[0][col]]
                lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
                c = 2 * asin(sqrt(a))
                r = 6371
                distance = (c * r)/100
                print(distance)
                if distance < 0:
                    distance = 0 - distance
                if distance != 0:
                    matrix[row][col] = 1/distance
                else:
                    matrix[row][col] = 100
    return matrix

def writeInfo(dict, file):
    with open(file, "w", encoding="utf-8") as f:
        for key in dict.keys():
            value = dict[key]
            f.write("{0}:{1}\n".format(key, value))
        f.close()

if __name__ == '__main__':
    # 获取所有合格的poilist
    poilist, total_poi_geo_dict = create_key(file_total_geo)
    # 获取签到记录的基本信息
    user_poi_dict, time_poi_dict, poi_geo_dict = readFile(file)        #,poilist

    # total_user_poi_dict, time_poi_dict, poi_geo_dict = readFile("../f_data/checkins_valid.txt") 
    # writeInfo(total_user_poi_dict,file_total_user_poi_dict) 

    # writeInfo(user_poi_dict, file_user_poi_dict)       #训练集用户访问poi
    # writeInfo(time_poi_dict, file_time_poi_dict)       #训练集poi访问时间
    # writeInfo(poi_geo_dict, file_geo_poi_dict)         #训练集poi地理位置信息

    formatdata = format_data(user_poi_dict)

    # print("我终于跑完了！累……")

    p1 = Process(target=select_func_run, args=("co_visit", poilist, formatdata, file_co_visit))
    # 进程1：训练集poi共现矩阵构建
    # p2 = Process(target=select_func_run, args=("visit_time", poilist, time_poi_dict,file_visit_time))
    # 进程2：训练集poi访问时间矩阵构建
    # p3 = Process(target=select_func_run, args=("poi_geo", poilist, total_poi_geo_dict, file_poi_geo_sim))
    # 进程3：所有poi地理相似矩阵构建

    p1.start()
    # p2.start()
    # p3.start()
    #
    p1.join()  # 等待进程停止，进程都停止以后才会执行接下来的语句
    # p2.join()
    # p3.join()
