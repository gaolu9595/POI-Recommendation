# 载入之前训练好的模型，来进行prediction

import numpy as np
from keras.models import load_model
from multiprocessing import Process
# import pickle
# from Network import My_class

def readFile(file):
    time_user_dict = {}
    poilist = []
    userlist = []
    for line in file.readlines():
        info = line.split("	")
        user_id = int(info[0])
        time = int(info[4][11:13])
        poi_id = int(info[5])
        if poi_id not in poilist:
            poilist.append(poi_id)
            # print("加入测试集poi：",poi_id)
        if user_id not in userlist:
            userlist.append(user_id)
            # print("加入测试集user：",user_id)
        if time not in time_user_dict.keys():
            time_user_dict[time] = [user_id]
        elif user_id not in time_user_dict[time]:
            time_user_dict[time].append(user_id)
    return poilist, userlist, time_user_dict

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
    for i in range(len(time_features[0,:])):
        key = i
        value = time_features[0:,i]
        time_dict[key] = value
        print("Join Time{0}:{1}".format(key, value))
    return time_dict

def get_user_visit_already(file_train_user_visit):
    user_visit_already = {}
    for line in file_train_user_visit.readlines():
        info = line.split(":")
        user = int(info[0])
        raw_pois = info[1].replace("[","").replace("]\n","").split(", ")
        pois = []
        for poi in raw_pois:
            poi = int(poi)
            pois.append(poi)
        user_visit_already[user] = pois
        print("New User{0}:{1}".format(user,pois))
    return user_visit_already

def __data_generation(user_emb, p_dict, poilist):
        '''Generates each batch data'''
        # 107*100 + 68 = 10768
        batch_size = 100
        batch_data = []
        # 开始格式化测试集
        j = 0
        step = 1
        for pid in poilist:
            value = np.concatenate((user_emb,p_dict[pid]),axis=0)
            if j < batch_size:
                batch_data.append(value)
                if j == 67 and step == 108:
                    yield np.array(batch_data)
                    break
                else:
                    j += 1
            else:
                yield np.array(batch_data)
                batch_data = [value]
                j = 1
                step += 1


def result_for_topk(time_user_dict,user_visit_already,u_dict,p_dict,userlist,poilist):
    my_dnn_model = load_model("./dnn_version/notime_twitter_dnn_trained_300dim_lapless_ver1.h5")
    k = 50      # Top_K推荐
    # userlist是测试集中的用户列表
    topk_user_poi = {}
    topk_pois = []
    for j in range(len(userlist)):
        user = userlist[j]
        user_visited = user_visit_already[user]
        user_emb = u_dict[user]
        print("Start Generating data……")
        rawresult = my_dnn_model.predict_generator(__data_generation(user_emb, p_dict, poilist), steps=int(len(poilist)/100)+1, verbose=1)
        # result是一个打分的numpy数组
        result = np.zeros((len(rawresult[0:,0]),2))
        print("=========",len(result[0:,0]),"===========")
        for num in range(len(result[0:,0])):
            result[num,0] = rawresult[num,0]
            result[num,1] = poilist[num]
        print(result)
        # 删除用户在训练集中已访问的poi
        length = len(result[0:,0])
        for num in range(length):
            if num < len(result[0:,0]):
                poi = int(result[num,1])
                if poi in user_visited:
                    print("删除poi{0}".format(poi))
                    result = np.delete(result,num,axis=0)
            else:
                break
        print("=========",len(result[0:,0]),"===========")
        # 按score大小排序[冒泡排序时间复杂度太高，直接调用numpy的排序函数即可]
        print("Start Sorting……")
        scores = result[:,0]
        sorted_scores = np.argsort(-scores)[:k]
        for index in sorted_scores:
            topk_pois.append(int(result[index,1]))
        print("Sorting Done……")
        topk_user_poi[user] = topk_pois
        topk_pois = []

    file_out = open("../t_data/test/recommendation/notime/300result_lapless.txt","w",encoding="utf-8")
    for user in userlist:
        if user in topk_user_poi.keys():
            file_out.write("{0}:{1}\n".format(user, topk_user_poi[user]))
    file_out.close()

if __name__ == '__main__':
    # 获取训练集中user已经访问的poi信息，以便在测试打分中删除
    file_train_user_visit = open("../t_data/train/t_train_user_visit.txt","r",encoding="utf-8")
    user_visit_already = get_user_visit_already(file_train_user_visit)
    file_train_user_visit.close()

    # 获取测试集中的poi\user数据，获取测试集中的签到正样本
    file_test = open("../t_data/t_test_set.txt","r",encoding="utf-8")
    poilist, userlist, time_user_dict = readFile(file_test)
    file_test.close()
    # print(poilist)

    # 获取user、poi以及time的嵌入
    # file_time = open("../g_data/matrix/g_smooth_train_time_sim_matrix.txt","r",encoding="utf-8")
    # time_dict = create_time_dict(file_time)
    # file_time.close()

    file_user = open("../t_data/embeddings/user/300user_embedding.csv","r",encoding="utf-8")
    file_poi = open("../t_data/embeddings/poi/300poi_embedding.csv","r",encoding="utf-8")
    # user和poi的embedding字典【是随着维度不同而变化的，在同一纬度下不随时间变化而变化】
    u_dict = create_emb_dict(file_user)
    p_dict = create_emb_dict(file_poi)
    emb_dim = 48+(4*300)

    result_for_topk(time_user_dict,user_visit_already,u_dict,p_dict,userlist,poilist)

    file_poi.close()
    file_user.close()

