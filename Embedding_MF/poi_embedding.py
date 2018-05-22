# 为每个poi（总15524）构建其embedding向量
# embedding的构建只来自于训练集的数据

import numpy as np

def readPOIS(file_pois):
    poilist = []
    with open(file_pois,"r",encoding="utf-8") as f:
        for line in f.readlines():
            poi_id = int(line.split(":")[0])
            if poi_id not in poilist:
                poilist.append(poi_id)
            print("FindNewPOI:{0}".format(poi_id))
        f.close()
    return poilist

def create_poi_embedding(file_covisit,file_geo,file_time,file_out,poilist):
    '''创建poi的特征表示'''
    # poi_embedding = {}
    covisit_matrix = np.loadtxt(open(file_covisit,"r",encoding="utf-8"),delimiter=",")
    geo_sim_matrix = np.loadtxt(open(file_geo,"r",encoding="utf-8"),delimiter=",")
    time_sim_matrix = np.loadtxt(open(file_time,"r",encoding="utf-8"),delimiter=",").T       # 取时间矩阵的转置
    time_sim_matrix = np.delete(time_sim_matrix,0,axis=0)
    time_sim_matrix = np.delete(time_sim_matrix,0,axis=1)

    # 特征矩阵的每一列都是一个poi向量的一部分，且三个矩阵的列都是严格对应的
    total_feature_matrix = np.concatenate((covisit_matrix,geo_sim_matrix,time_sim_matrix),axis=0)
    # 在features矩阵上加上一行poi的id表示，方便后续处理
    total_feature_matrix = np.insert(total_feature_matrix, 0, values=poilist, axis=0)
    print(total_feature_matrix)
    np.savetxt(file_out,total_feature_matrix,delimiter=",",fmt="%f")
    print("poi向量维度：",len(total_feature_matrix[:,0]))
    print("poi向量个数：",len(total_feature_matrix[0,:]))


if __name__ == '__main__':
    file_pois = "../data_5months/valid_total/g_valid_total_poi_geo.txt"
    poilist = readPOIS(file_pois)
    file_poi_visit_time = "../data_5months/train/g_train_time_sim_matrix.txt"
    dimension_list = [20,40,60,80,100,120]
    for dim in dimension_list:
        print("======================Start {0} Features Task============================".format(dim))
        file_poi_covisit = "../data_5months/train_feature_matrixs/covisit/{0}poi_covisit_feature.csv".format(dim)
        file_poi_geo = "../data_5months/valid_total_feature_matrixs/geo/{0}poi_geo_feature.csv".format(dim)
        file_out = "../data_5months/embeddings/poi/{0}poi_embedding.csv".format(dim)
        create_poi_embedding(file_poi_covisit,file_poi_geo,file_poi_visit_time,file_out,poilist)
