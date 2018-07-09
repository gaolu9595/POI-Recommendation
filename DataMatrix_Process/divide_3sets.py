#将有效的签到记录数据划分为训练集、测试集和参数调优集
#保证：每个集合中每个用户都是至少有一条签到记录的
import  random
import numpy as np

#file_matrix = "../data/g_time_sim_matirx.txt"
file_valid_total = "../data/Gowalla_Valid_Data.txt"

#从总签到数据中筛选出每个用户的70%作为训练集，20%作为测试集，10%作为参数调优集
def divide_train_test_tune(file):
    with open(file,"r",encoding="utf-8") as f:
        user_total = {}
        #将每个用户的签到记录保存在dict中
        for line in f.readlines():
            uid = line.split("	")[0]
            if uid in user_total.keys():
                user_total[uid].append(line)
            else:
                user_total[uid] = [line]
                print("收集用户{0}签到记录".format(uid))
    #随机选取和分配三个集合的元素,写入相应的文件中
    f1 = open("../data/g_train_set.txt","w",encoding="utf-8")
    f2 = open("../data/g_test_set.txt","w",encoding="utf-8")
    f3 = open("../data/g_tune_set.txt","w",encoding="utf-8")
    #将控制台输出到文件
    f4 = open("../bugtest/promise_validation.txt","w",encoding="utf-8")
    for user in user_total.keys():
        #records是user的签到记录（str）
        records = user_total[user]
        poilist = []
        poi_record = {}
        for record in records:
            poi = int(record.split("	")[4].strip("\n"))
            if poi not in poilist:
                poilist.append(poi)
            if poi not in poi_record.keys():
                poi_record[poi] = [record]
            else:
                poi_record[poi].append(record)
        print(len(poilist))
        #确定不同的poi总数，划分数目
        record_num = len(poilist)
        tune_num = int(record_num*0.1)
        test_num = int(record_num*0.2)
        train_num = record_num - tune_num - test_num

        poilist1 = random.sample(poilist,train_num)                 #随机采样的函数,按poi比例采样
        #对于采样到的poi，其被访问记录写入文件
        for poi_id in poilist1:
            for line in poi_record[poi_id]:
                f1.write(line)
        #list1 = random.sample(records,train_num)
        print("User {0}-------train set(size:{1},data:{2}".format(user,train_num,poilist1))
        f4.write("User {0}-------train set(size:{1},data:{2}\n".format(user,train_num,poilist1))
        # for line in list1:
        #     f1.write(line)
        condition1 = lambda c:c not in poilist1                   #筛选符合条件的列表元素

        filter_train_pois = list(filter(condition1,poilist))
        poilist2 = random.sample(filter_train_pois,test_num)
        # filter_train_records = list(filter(condition1,records))
        # list2 = random.sample(filter_train_records,test_num)
        for poi_id in poilist2:
            for line in poi_record[poi_id]:
                f2.write(line)
        print("User {0}-------test set(size:{1},data:{2}".format(user,test_num,poilist2))
        f4.write("User {0}-------test set(size:{1},data:{2}\n".format(user,test_num,poilist2))
        # for line in list2:
        #     f2.write(line)
        condition2 = lambda c:c not in poilist2

        poilist3 = list(filter(condition2,filter_train_pois))
        for poi_id in poilist3:
            for line in poi_record[poi_id]:
                f3.write(line)
        print("User {0}-------tune set(size:{1},data:{2}".format(user,tune_num,poilist3))
        f4.write("User {0}-------tune set(size:{1},data:{2}\n".format(user,tune_num,poilist3))
        # for line in list3:
        #     f3.write(line)

if __name__ == '__main__':
    divide_train_test_tune(file_valid_total)
