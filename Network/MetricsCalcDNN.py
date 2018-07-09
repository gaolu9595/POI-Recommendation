# 针对测试集的topk推荐结果和groundtruth，求解出评价标准

def calcHitNum(file_groundtruth,file_recommendation):
    true_user_pois = {}
    rec_user_pois = {}
    userlist = []
    hit_count = {}
    for line in file_groundtruth:
        info = line.split(":")
        user = int(info[0])
        pois = info[1].replace("[","").replace("]\n","").split(", ")
        for poi in pois:
            poi = int(poi)
        true_user_pois[user] = pois
        userlist.append(user)
    for line in file_recommendation:
        info = line.split(":")
        user = int(info[0])
        pois = info[1].replace("[","").replace("]\n","").split(", ")
        for poi in pois:
            poi = int(poi)
        rec_user_pois[user] = pois[:k]
    for user in userlist:
        count = 0
        for poi in rec_user_pois[user]:
            if poi in true_user_pois[user]:
                count += 1
        hit_count[user] = count
    return userlist,hit_count

def calcPrecision(userlist, hit_count, k):
    user_num = len(userlist)
    precision = 0
    for user in userlist:
        precision += float(hit_count[user]/k)
    return float(precision/user_num)

def calcRecall(userlist, hit_count, file_user_poinum):
    user_num = len(userlist)
    user_poinum = {}
    recall = 0
    for line in file_user_poinum:
        info = line.split(":")
        user = int(info[0])
        poinum = int(info[1])
        user_poinum[user] = poinum
    for user in userlist:
        recall += float(hit_count[user]/user_poinum[user])
    return float(recall/user_num)

if __name__ == '__main__':
    timelist = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    # 测试
    dimlist = [300]
    k = 5
    precision_each_dim = {}
    recall_each_dim = {}
    for dim in dimlist:
        precision = {}
        recall = {}
        precision_each_dim[dim] = 0
        recall_each_dim[dim] = 0
        # for time in timelist:
            # print("============Time {0}=============".format(time))
            # file_groundtruth = open("../t_data/test/groundtruth/{0}positive.txt".format(time),"r",encoding="utf-8")
            # file_user_poinum = open("../t_data/test/groundtruth/{0}positive_usernum.txt".format(time),"r",encoding="utf-8")
            # file_recommendation = open("../t_data/test/recommendation/ver1/{0}result_{1}time.txt".format(dim,time),"r",encoding="utf-8")

        # for time in timelist:
            # print("============Time {0}=============".format(time))
            # file_groundtruth = open("../g_data/test/groundtruth/{0}positive.txt".format(time),"r",encoding="utf-8")
            # file_user_poinum = open("../g_data/test/groundtruth/{0}positive_usernum.txt".format(time),"r",encoding="utf-8")
            # file_recommendation = open("../g_data/test/recommendation/ver1/{0}result_{1}time.txt".format(dim,time),"r",encoding="utf-8")

        for time in timelist:
            print("============Time {0}=============".format(time))
            file_groundtruth = open("../f_data/test/groundtruth/{0}positive.txt".format(time),"r",encoding="utf-8")
            file_user_poinum = open("../f_data/test/groundtruth/{0}positive_usernum.txt".format(time),"r",encoding="utf-8")
            file_recommendation = open("../f_data/test/recommendation/f_threshold_bn_uniform/{0}result_{1}time.txt".format(dim,time),"r",encoding="utf-8")
            # 计算命中数目
            print("Calc Hit Count……")
            userlist, hit_count = calcHitNum(file_groundtruth,file_recommendation)
            # 计算该时间内所有用户的总准确率和总召回率
            print("Calc Precision and Recall……")
            precision[time] = calcPrecision(userlist, hit_count, k)
            recall[time] = calcRecall(userlist, hit_count, file_user_poinum)

            precision_each_dim[dim] += precision[time]
            recall_each_dim[dim] += recall[time]
            file_groundtruth.close()
            file_user_poinum.close()
            file_recommendation.close()
        # 计算该维度内所有时间的总准确率和总召回率
        precision_each_dim[dim] /= 24
        recall_each_dim[dim] /= 24

        print("precision dim{0}:{1}".format(dim,precision_each_dim[dim]))
        print("recall dim{0}:{1}".format(dim,recall_each_dim[dim]))

