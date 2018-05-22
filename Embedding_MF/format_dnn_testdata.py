# 将测试集的格式标准化，化成适合深度学习输入的embedding格式
# Data Format: user_emb + poi_emb(test_set) + time_emb, score
# 数据量很大: 7023*13040*24 = 22亿的数量级     # 2197918080
# 按时间来分开存储。例：在时间点t签到的用户数为n，则时间点t的测试条目数有t*n*13040个
import numpy as np

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
            print("加入测试集poi：",poi_id)
        if user_id not in userlist:
            userlist.append(user_id)
            print("加入测试集user：",user_id)
        if time not in time_user_dict.keys():
            time_user_dict[time] = [user_id]
        elif user_id not in time_user_dict[time]:
            time_user_dict[time].append(user_id)
            print("加入时间{0}中签到的user{1}".format(time, user_id))
        # checkin_info.append([user_id,time,poi_id])
    print("测试集poi总数:",len(poilist))
    print("测试集user总数:",len(userlist))
    # print("测试集签到记录总数：",len(checkin_info))         # 71689
    time_file = open("../data_5months/test_format_input/time_user_dict.txt","w",encoding="utf-8")
    for time in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]:
        time_file.write("{0}:{1}\n".format(time,time_user_dict[time]))
    print("保存测试集时间用户列表Done！")
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

def create_formatdata(u_dict, p_dict, time_emb, poilist, time_userlist, file_out, emb_dim, size):
    # 开始格式化测试集
    j = 0
    poisize = len(poilist)
    emb_matrix = np.zeros((emb_dim,size))
    print("矩阵尺寸({0},{1})".format(emb_dim,size))
    for uid in time_userlist:
        for pid in poilist:
            emb_matrix[:,j] = np.concatenate((u_dict[uid],p_dict[pid],time_emb),axis=0)
            if (j % poisize) == 0:
                print("Test_Emb_Format {0}:{1}".format(j, emb_matrix[:,j]))
            j += 1
    print(j)
    np.savetxt(file_out,emb_matrix,delimiter=",",fmt="%f")

if __name__ == '__main__':
    # 获取测试集中的poi\user数据，获取测试集中的签到正样本
    file_test = open("../data_5months/g_test_set.txt","r",encoding="utf-8")
    poilist, userlist, time_user_dict = readFile(file_test)
    file_test.close()

    # 获取user、poi以及time的嵌入
    file_time = open("../data_5months/train/g_smooth_train_time_sim_matrix.csv","r",encoding="utf-8")
    time_dict = create_time_dict(file_time)
    file_time.close()

    # 格式化测试数据
    # dim_list = [20]
    dim_list = [20,40,60,80,100,120]
    timelist = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    for i in dim_list:
        print("=======================Start {0} Dimension Creation==========================".format(i))
        file_user = open("../data_5months/embeddings/user/{0}user_embedding.csv".format(i),"r",encoding="utf-8")
        file_poi = open("../data_5months/embeddings/poi/{0}poi_embedding.csv".format(i),"r",encoding="utf-8")
        # user和poi的embedding字典【是随着维度不同而变化的，在同一纬度下不随时间变化而变化】
        u_dict = create_emb_dict(file_user)
        p_dict = create_emb_dict(file_poi)
        emb_dim = 72+(5*i)
        for j in timelist:
            file_out = open("../data_5months/test_format_input/g_test_format_{0}embedding_{1}time.csv".format(i,j),"w",encoding="utf-8")
            time_userlist = time_user_dict[j]
            num = len(time_userlist)
            poisize = len(poilist)
            size = num*poisize
            print(size)
            time_emb = time_dict[j]
            print("=====================Time {0} Matrix======================".format(j))
            create_formatdata(u_dict, p_dict, time_emb, poilist, time_userlist, file_out, emb_dim, size)
            file_out.close()
            file_poi.close()
            file_user.close()
