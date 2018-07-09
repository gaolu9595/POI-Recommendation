# 将训练集、验证集的格式标准化，化成适合深度学习输入的embedding格式
# Foursquare数据集
# Data Format: user_emb+poi_emb+time_sim(tensor), label(0/1)
# 是以Numpy矩阵的格式读入网络进行操作的

import numpy as np
from random import choice
from sklearn import preprocessing

def create_emb_dict(file):
    '''获取User和POI的嵌入'''
    emb_dict = {}
    features = np.loadtxt(file,delimiter=",")
    for i in range(len(features[0,:])):
        key = int(features[0,i])
        value = features[1:,i]
        emb_dict[key] = value
        if i == 0:
            print("Join {0}:{1}".format(key,value))
    print(len(emb_dict))
    return emb_dict

def create_time_dict(file_time):
    '''创建时间emb'''
    time_dict = {}
    time_features =np.loadtxt(file_time,delimiter=",")
    time_features =preprocessing.scale(time_features,axis=1)
    for i in range(len(time_features[0,:])):
        key = i
        value = time_features[0:,i]
        time_dict[key] = value
        print("Join Time{0}:{1}".format(key, value))
    return time_dict

def readUserVisit(file_user_visit):
    '''创建用户访问poi记录'''
    user_visit_dict = {}
    train_poilist = []
    count = 1
    for line in file_user_visit.readlines():
        info = line.split(":")
        user = int(info[0])
        raw_pois = info[1].replace("[","").replace("]\n","").split(", ")
        pois = []
        for poi in raw_pois:
            poi = int(poi)
            pois.append(poi)
            if poi not in train_poilist:
                train_poilist.append(poi)
                # print("New POI{0}:{1}".format(count,poi))
                count += 1
        user_visit_dict[user] = pois
        # print("New User{0}:{1}".format(user,pois))
    return user_visit_dict,train_poilist

# 根据签到记录，获取网络输入的正样本，最后一行是样本标签（0/1）
# 随机抽样产生负样本(10,15,20,25,30),或者对每一条观测到的正样本随机产生一条与之对应的负样本
def format_train_checkins(file_train, u_dict, p_dict, time_dict, user_visit_dict, train_poilist, file_out, emb_dim):
    # 开始格式化签到记录
    j = 0
    # 304047*2
    pos_emb_matrix = np.zeros((emb_dim,151589))
    neg_emb_matrix = np.zeros((emb_dim,151589))
    timelist = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    user_negative_pois = {}
    for uid in user_visit_dict.keys():
        condition = lambda l: l not in user_visit_dict[uid]
        negative_pois = list(filter(condition,train_poilist))
        user_negative_pois[uid] = negative_pois
    for line in file_train.readlines():
        info = line.split("	")
        uid = int(info[0][5:])
        time = int(info[3][0:2])
        pid = int(info[1][4:])
        if j < 151589:
            pos_emb_matrix[0:,j] = np.concatenate((u_dict[uid],p_dict[pid],time_dict[time]),axis=0)
            if j == 0 or j == 151588:
                print("Positive_Format {0}:{1}".format(j, pos_emb_matrix[0:,j]))
            sample_poi = choice(user_negative_pois[uid])      # 从负例中随机获取一个样本
            # print(sample_poi)
            # sample_time = np.random.choice(timelist,1)
            neg_emb_matrix[0:,j] = np.concatenate((u_dict[uid],p_dict[sample_poi],time_dict[time]),axis=0)
            if j == 0 or j == 151588:
                print("Negative_Format {0}:{1}".format(j, neg_emb_matrix[0:,j]))
            j += 1
    emb_matrix = np.concatenate((pos_emb_matrix,neg_emb_matrix),axis=1)
    np.savetxt(file_out,emb_matrix,delimiter=",",fmt="%f")

def format_tune_checkins(file_tune, u_dict, p_dict, time_dict, file_out, emb_dim):
    # 开始格式化签到记录
    j = 0
    # 33359
    emb_matrix = np.zeros((emb_dim, 10778))
    for line in file_tune.readlines():
        info = line.split("	")
        uid = int(info[0][5:])
        time = int(info[3][0:2])
        pid = int(info[1][4:])
        emb_matrix[0:,j] = np.concatenate((u_dict[uid],p_dict[pid],time_dict[time]),axis=0)
        if j == 0 or j == 10777:
            print("Tune_Format {0}:{1}".format(j, emb_matrix[0:,j]))
        j += 1
    np.savetxt(file_out,emb_matrix,delimiter=",",fmt="%f")

if __name__ == '__main__':
    # file_checkin = open("../data_5months/g_train_set.txt","r",encoding="utf-8")
    # file_checkin需要放到循环里面去，因为第一次读文件读到最后一列时游标就停在最后一列，只要没有关闭该文件，游标就一直在下面
    # 所以在循环第二次时没办法再读这个checkin文件了
    # 写代码真的还有很长一段路要走啊
    file_time = open("../f_data/matrix/f_smooth_train_time_sim_matrix.txt","r",encoding="utf-8")
    file_user_visit = open("../f_data/train/f_train_user_visit.txt","r",encoding="utf-8")
    # 以下的信息只需要构建一次【随着维度变化，时间emb不变】
    time_dict = create_time_dict(file_time)
    user_visit_dict, train_poilist = readUserVisit(file_user_visit)
    # print(len(train_poilist))
    # print(len(user_visit_dict))
    file_time.close()
    file_user_visit.close()

    # 格式化签到信息，以下信息需要多次构建
    dim_list = [100,200,300]
    for i in dim_list:
        print("=======================Start {0} Dimension Creation==========================".format(i))
        # file_train = open("../f_data/f_train_set.txt","r",encoding="utf-8")
        file_tune = open("../f_data/f_tune_set.txt","r",encoding="utf-8")
        file_user = open("../f_data/embeddings/user/neighbor_norm/{0}user_embedding.csv".format(i),"r",encoding="utf-8")
        file_poi = open("../f_data/embeddings/poi/neighbor_norm/{0}poi_embedding.csv".format(i),"r",encoding="utf-8")
        # file_out1 = open("../f_data/train_format_input/neighbor_norm/f_train_format_{0}embedding.csv".format(i),"w",encoding="utf-8")
        file_out2 = open("../f_data/tune_format_input/neighbor_norm/f_tune_format_{0}embedding.csv".format(i),"w",encoding="utf-8")
        # user和poi的embedding字典【是随着维度不同而变化的】
        u_dict = create_emb_dict(file_user)
        p_dict = create_emb_dict(file_poi)
        emb_dim = 72+(4*i)
        # format_train_checkins(file_train, u_dict, p_dict, time_dict, user_visit_dict, train_poilist, file_out1, emb_dim)
        format_tune_checkins(file_tune, u_dict, p_dict, time_dict, file_out2, emb_dim)
        file_out2.close()
        # file_out1.close()
        file_poi.close()
        file_user.close()
        # file_train.close()
        file_tune.close()

