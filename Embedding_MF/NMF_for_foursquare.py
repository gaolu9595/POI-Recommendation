# 对Gowalla数据集构造的poi的共现矩阵、地理位置矩阵进行NMF
# NMF维度：50、100、150、200、250、300

import numpy as np
from multiprocessing import Process
from sklearn.decomposition import NMF

def readFile(file_matrix):
    '''读取矩阵文件，获得user列表，获得矩阵信息'''
    print("=====================Start Reading Matrix==========================")
    matrix = np.loadtxt(open(file_matrix,"r",encoding="utf-8"),delimiter=",")
    print("=====================Start Collecting Users==========================")
    itemlist = list(matrix[1:,0])
    for item in itemlist:
        item = int(item)
        # print(item)
    matrix = np.delete(matrix,0,axis=0)
    matrix = np.delete(matrix,0,axis=1)
    print(matrix)
    return itemlist, matrix

def doNMF(matrix,type,dim):
    '''对矩阵进行NMF操作'''
    # nmf_model = NMF(n_components=dim,init="nndsvd",solver="cd",max_iter=500,verbose=1)
    nmf_model = NMF(n_components=dim,init="nndsvd",solver="cd",alpha=0.001,l1_ratio=0,verbose=1,max_iter=3000)
    print("=====================Start Doing NMF==========================")
    baseVectors = nmf_model.fit_transform(matrix)
    featureVectors = nmf_model.components_
    loss = nmf_model.reconstruction_err_
    num_iter = nmf_model.n_iter_
    print("=================W：基矩阵===================")
    print(baseVectors)
    print("=================H：特征矩阵===================")
    print(featureVectors)
    print("{0}矩阵由{1}个向量组成".format(dim,len(featureVectors[0,:])))
    # file_info_out.write("{0}矩阵由{1}个向量组成\n".format(dim,len(featureVectors[0,:])))
    if type == 0:
        np.savetxt("../f_data/nmf_result/social/{0}user_social_feature.csv".format(dim),featureVectors,delimiter=",",fmt="%f")
    elif type == 1:
        np.savetxt("../f_data/nmf_result/geo/threshold/{0}poi_geo_feature_l2penalty.csv".format(dim),featureVectors,delimiter=",",fmt="%f")
    else:
        np.savetxt("../f_data/nmf_result/covisit/{0}poi_covisit_feature_l2penalty.csv".format(dim),featureVectors,delimiter=",",fmt="%f")
    print("原矩阵与{0}WH的差异：{1}；迭代总次数：{2}".format(dim,loss,num_iter))
    # file_info_out.write("原矩阵与{0}WH的差异：{1}；迭代总次数：{2}\n".format(dim,loss,num_iter))
    return loss,num_iter

if __name__ == '__main__':
    '''对训练集共现矩阵、总poi地理矩阵、总user邻居矩阵做NMF【训练集poi访问时间矩阵，不需要做降维目的的NMF】'''
    # 矩阵文件,读取矩阵信息,做非负矩阵分解
    # file_social_matrix = "../g_data/matrix/g_total_user_social_matrix.txt"
    # userlist, user_social_matrix = readFile(file_social_matrix)

    file_poi_geo_matrix = "../f_data/matrix/bugtest_geosim_threshold4.txt"
    poilist, poi_geo_matrix = readFile(file_poi_geo_matrix)


    # file_poi_covisit_matrix = "../f_data/matrix/bugtest_covisit.txt"
    # poilist, poi_covisit_matrix = readFile(file_poi_covisit_matrix)


    p1 = Process(target=doNMF, args=(poi_geo_matrix,1,100))
    p2 = Process(target=doNMF, args=(poi_geo_matrix,1,200))
    p3 = Process(target=doNMF, args=(poi_geo_matrix,1,300))
    # p4 = Process(target=doNMF, args=(poi_covisit_matrix,2,100))
    # p5 = Process(target=doNMF, args=(poi_covisit_matrix,2,200))
    # p6 = Process(target=doNMF, args=(poi_covisit_matrix,2,300))

    p1.start()
    p2.start()
    p3.start()
    # p4.start()
    # p5.start()
    # p6.start()
 
    p1.join()  # 等待进程停止，进程都停止以后才会执行接下来的语句
    p2.join()
    p3.join()
    # p4.join()
    # p5.join()
    # p6.join()





