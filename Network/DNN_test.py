# 载入之前训练好的模型，来进行prediction

import numpy as np
from keras.models import load_model
from multiprocessing import Process

from Network import My_class

def readFile(file):
    time_user_dict = {}
    poilist = []
    userlist = []
    # checkin_info = []
    # test_poi_geo_dict = {}
    for line in file.readlines():
        info = line.split("	")
        user_id = int(info[0])
        time = int(info[1][11:13])
        poi_id = int(info[4])
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
            # print("加入时间{0}中签到的user{1}".format(time, user_id))
        # checkin_info.append([user_id,time,poi_id])
    # print("测试集poi总数:",len(poilist))
    # print("测试集user总数:",len(userlist))
    # poi_file = open("../data_5months/test_format_input/test_poilist.txt","w",encoding="utf-8")
    # for poi in poilist:
    #     poi_file.write("{0}\n".format(poi))
    # print("保存测试集时间用户列表Done！")
    # print("测试集签到记录总数：",len(checkin_info))         # 71689
    # time_file = open("../data_5months/test_format_input/time_user_dict.txt","w",encoding="utf-8")
    # for time in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]:
    #     time_file.write("{0}:{1}\n".format(time,time_user_dict[time]))
    # print("保存测试集时间用户列表Done！")
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

def __data_generation(user_emb, p_dict, time_emb, poilist):
        '''Generates each batch data'''
        batch_size = 40
        batch_data = []
        # 开始格式化测试集
        j = 0
        step = 1
        for pid in poilist:
            value = np.concatenate((user_emb,p_dict[pid],time_emb),axis=0)
            if j < batch_size:
                batch_data.append(value)
                if j == 39 and step == 326:
                    # print("Batch {0}: Last Record{1}".format(step,value))
                    yield np.array(batch_data)
                    break
                else:
                    j += 1
            else:
                # print("Batch {0}: Last Record{1}".format(step,batch_data[-1]))
                yield np.array(batch_data)
                batch_data = [value]
                j = 1
                step += 1

def result_for_topk(timelist,time_dict,time_user_dict,user_visit_already,u_dict,p_dict,userlist,poilist):
    my_dnn_model = load_model("dnn_trained_20dim_4km.h5")
    k = 50      # Top_K推荐
    for i in timelist:
        print("===================Time {0}======================".format(i))
        time_emb = time_dict[i]
        time_userlist = time_user_dict[i]
        size = 2
        # 对时间i内每个用户的top k个签到poi进行记录
        topk_user_poi = {}
        topk_pois = []
        for j in range(size):
            user = time_userlist[j]
            user_visited = user_visit_already[user]
            user_emb = u_dict[user]
            # 当generator是Sequence的一个实例时，steps可以为none        # batch_size为40时，len为整数
            # batch_testdata_generator = My_class.DataGenerator(user_emb, p_dict, time_emb, poilist)
            print("Start Generating data……")
            rawresult = my_dnn_model.predict_generator(__data_generation(user_emb, p_dict, time_emb, poilist), steps=int(len(poilist)/40), verbose=1)
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
            # for m in range(length-1):
            #     for n in range(length-1-m):
            #         if result[n,0] < result[n+1,0]:
            #             max_score = result[n+1,0]
            #             max_poi = result[n+1,1]
            #             result[n+1,0] = result[n,0]
            #             result[n+1,1] = result[n,1]
            #             result[n,0] = max_score
            #             result[n,1] = max_poi
            topk_user_poi[user] = topk_pois
            topk_pois = []
        file_out = open("../data_5months/test/20result_{0}time.txt".format(i),"w",encoding="utf-8")
        for user in userlist:
            if user in topk_user_poi.keys():
                file_out.write("{0}:{1}\n".format(user, topk_user_poi[user]))
        file_out.close()

if __name__ == '__main__':
    # 获取训练集中user已经访问的poi信息，以便在测试打分中删除
    file_train_user_visit = open("../data_5months/train/g_train_user_visit.txt","r",encoding="utf-8")
    user_visit_already = get_user_visit_already(file_train_user_visit)
    file_train_user_visit.close()

    # 获取测试集中的poi\user数据，获取测试集中的签到正样本
    file_test = open("../data_5months/g_test_set.txt","r",encoding="utf-8")
    poilist, userlist, time_user_dict = readFile(file_test)
    file_test.close()

    # 获取user、poi以及time的嵌入
    file_time = open("../data_5months/train/g_smooth_train_time_sim_matrix.csv","r",encoding="utf-8")
    time_dict = create_time_dict(file_time)
    file_time.close()

    file_user = open("../data_5months/embeddings/user/20user_embedding.csv","r",encoding="utf-8")
    file_poi = open("../data_5months/embeddings/poi/20poi_embedding.csv","r",encoding="utf-8")
    # user和poi的embedding字典【是随着维度不同而变化的，在同一纬度下不随时间变化而变化】
    u_dict = create_emb_dict(file_user)
    p_dict = create_emb_dict(file_poi)
    emb_dim = 72+(5*20)

    # 对于一个给定时间，计算该时间内有签到的用户与所有poi的签到可能性
    # timelist = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    timelist1 = [0,1,2]
    timelist2 = [3,4,5]
    timelist3 = [6,7,8]
    timelist4 = [9,10,11]
    timelist5 = [12,13,14]
    timelist6 = [15,16,17]
    timelist7 = [18,19,20]
    timelist8 = [21,22,23]
    # 多进程传输的参数必须是可序列化的
    # pool进程池可以直接自动分配多进程，并得到结果
    # pool = Pool()
    # result = pool.map(result_for_topk)
    p1 = Process(target=result_for_topk, args=(timelist1,time_dict,time_user_dict,user_visit_already,u_dict,p_dict,userlist,poilist))
    p2 = Process(target=result_for_topk, args=(timelist2,time_dict,time_user_dict,user_visit_already,u_dict,p_dict,userlist,poilist))
    p3 = Process(target=result_for_topk, args=(timelist3,time_dict,time_user_dict,user_visit_already,u_dict,p_dict,userlist,poilist))
    p4 = Process(target=result_for_topk, args=(timelist4,time_dict,time_user_dict,user_visit_already,u_dict,p_dict,userlist,poilist))
    p5 = Process(target=result_for_topk, args=(timelist5,time_dict,time_user_dict,user_visit_already,u_dict,p_dict,userlist,poilist))
    p6 = Process(target=result_for_topk, args=(timelist6,time_dict,time_user_dict,user_visit_already,u_dict,p_dict,userlist,poilist))
    p7 = Process(target=result_for_topk, args=(timelist7,time_dict,time_user_dict,user_visit_already,u_dict,p_dict,userlist,poilist))
    p8 = Process(target=result_for_topk, args=(timelist8,time_dict,time_user_dict,user_visit_already,u_dict,p_dict,userlist,poilist))
    p1.start()   # 启动进程
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()

    p1.join()  # 等待进程停止
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()

    file_poi.close()
    file_user.close()
