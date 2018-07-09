# 对Gowalla数据集构造的poi的共现矩阵、地理位置矩阵进行NMF
# NMF维度：50、100、150、200、250、300

import numpy as np
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
    nmf_model = NMF(n_components=dim,init="nndsvd",solver="cd",max_iter=5000,verbose=1,)
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
        np.savetxt("../g_data/nmf_result/social/{0}user_social_feature.csv".format(dim),featureVectors,delimiter=",",fmt="%f")
    elif type == 1:
        np.savetxt("../g_data/nmf_result/geo/{0}poi_geo_feature.csv".format(dim),featureVectors,delimiter=",",fmt="%f")
    else:
        np.savetxt("../g_data/nmf_result/covisit/{0}poi_covisit_feature.csv".format(dim),featureVectors,delimiter=",",fmt="%f")
    print("原矩阵与{0}WH的差异：{1}；迭代总次数：{2}".format(dim,loss,num_iter))
    # file_info_out.write("原矩阵与{0}WH的差异：{1}；迭代总次数：{2}\n".format(dim,loss,num_iter))
    return loss,num_iter

if __name__ == '__main__':
    '''对训练集共现矩阵、总poi地理矩阵、总user邻居矩阵做NMF【训练集poi访问时间矩阵，不需要做降维目的的NMF】'''
    # 矩阵文件,读取矩阵信息,做非负矩阵分解
    # file_social_matrix = "../g_data/matrix/g_total_user_social_matrix.txt"
    # userlist, user_social_matrix = readFile(file_social_matrix)
    # loss1,num_iter1 = doNMF(user_social_matrix,0,50)
    # loss2,num_iter2 = doNMF(user_social_matrix,0,100)
    # loss3,num_iter3 = doNMF(user_social_matrix,0,150)
    # loss4,num_iter4 = doNMF(user_social_matrix,0,200)
    # loss5,num_iter5 = doNMF(user_social_matrix,0,250)
    # loss6,num_iter6 = doNMF(user_social_matrix,0,300)
    # social_loss_list = [loss1,loss2,loss3,loss4,loss5,loss6]
    # social_numiter_list = [num_iter1,num_iter2,num_iter3,num_iter4,num_iter5,num_iter6]

    file_poi_geo_matrix = "../g_data/matrix/g_valid_total_poi_sim_matrix.txt"
    poilist, poi_geo_matrix = readFile(file_poi_geo_matrix)
    # loss1,num_iter1 = doNMF(poi_geo_matrix,1,50)
    # loss2,num_iter2 = doNMF(poi_geo_matrix,1,100)
    # loss3,num_iter3 = doNMF(poi_geo_matrix,1,150)
    # loss4,num_iter4 = doNMF(poi_geo_matrix,1,200)
    # loss5,num_iter5 = doNMF(poi_geo_matrix,1,250)
    loss6,num_iter6 = doNMF(poi_geo_matrix,1,300)
    # geo_loss_list = [loss1,loss2,loss3,loss4,loss5,loss6]
    # geo_numiter_list = [num_iter1,num_iter2,num_iter3,num_iter4,num_iter5,num_iter6]

    # file_poi_covisit_matrix = "../g_data/matrix/g_train_co_visit_matrix.txt"
    # poilist, poi_covisit_matrix = readFile(file_poi_covisit_matrix)
    # loss1,num_iter1 = doNMF(poi_covisit_matrix,2,50)
    # loss2,num_iter2 = doNMF(poi_covisit_matrix,2,100)
    # loss3,num_iter3 = doNMF(poi_covisit_matrix,2,150)
    # loss4,num_iter4 = doNMF(poi_covisit_matrix,2,200)
    # loss5,num_iter5 = doNMF(poi_covisit_matrix,2,250)
    # loss6,num_iter6 = doNMF(poi_covisit_matrix,2,300)
    # covisit_loss_list = [loss1,loss2,loss3,loss4,loss5,loss6]
    # covisit_numiter_list = [num_iter1,num_iter2,num_iter3,num_iter4,num_iter5,num_iter6]



