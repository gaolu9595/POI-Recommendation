# 处理Twitter数据集中的数据
# 删除不合法的user和poi
# 划分训练集、调优集和测试集
import random

file_source = "../twitter/checkins.txt"
file_target = "../twitter/checkins_valid.txt"
file_valid_user_visit = "../t_data/valid_total/t_valid_total_user_visit.txt"
file_valid_poi_list = "../t_data/valid_total/t_valid_total_poi_geo.txt"
file_valid_visie_time = "../t_data/valid_total/t_valid_total_visit_time.txt"

def filter_user_poi(file_total):
    with open(file_total,"r",encoding="utf-8") as f:
        user_poi_dict = {}
        time_poi_dict = {}
        poi_geo_dict = {}
        for line in f.readlines():
            info = line.split("	")
            user_id = int(info[0])
            time = info[4]
            poi_id = int(info[5])
            geo = [float(info[2]),float(info[3])]
            #构建用户访问poi列表的字典
            if user_id in user_poi_dict.keys():
                if poi_id not in user_poi_dict[user_id]:      #保证一个user访问过的每个poi只出现一次
                    user_poi_dict[user_id].append(poi_id)
            else:
                user_poi_dict[user_id] = [poi_id]
                print("User",user_id)
            #对时间格式进行预处理,获取签到行为的时间点(24小时制)
            time = int(time[11:13])
            #构建poi被访问时间的字典，只选取部分时间内的poi签到记录
            if time in time_poi_dict.keys():
                #每个时间的poi列表是有重复元素的
                time_poi_dict[time].append(poi_id)          #一个时间内的poi可以出现多次
            else:
                time_poi_dict[time] = [poi_id]
            #构建poi地理位置字典,只选取美国本土的poi
            if poi_id not in poi_geo_dict.keys():
                poi_geo_dict[poi_id] = [geo,[user_id]]
            elif user_id not in poi_geo_dict[poi_id][1]:
                poi_geo_dict[poi_id][1].append(user_id)

        print("====每个poi的地理位置经纬度信息====")
        for poi in list(poi_geo_dict.keys()):
            #filter签到用户数小于10的poi
            if len(poi_geo_dict[poi][1]) < 10:
                poi_geo_dict.pop(poi)
            else:
                print(poi,":",poi_geo_dict[poi])
                poi_geo_dict[poi] = poi_geo_dict[poi][0]
                print(poi,":",poi_geo_dict[poi])

        valid_userlist = []
        valid_poilist = list(poi_geo_dict.keys())            #有效的poilist
        print("有效POI列表：",valid_poilist)
        print("长度：",len(valid_poilist))
        condition = lambda c: c in valid_poilist

        print("====每个用户访问过的poi列表====")
        for user in list(user_poi_dict.keys()):
            #filter签到poi少于15个的用户
            current_poilist = user_poi_dict[user]
            user_poi_dict[user] = list(filter(condition,current_poilist))
            if len(user_poi_dict[user]) < 15:
                user_poi_dict.pop(user)
            else:
                print(user,":",user_poi_dict[user])
                valid_userlist.append(user)
        print("有效user列表：{0}".format(valid_userlist))
        print("====每个poi的访问时间信息====")
        for time in list(time_poi_dict.keys()):
            #filter时间字典中不合格的poi
            current_poilist = time_poi_dict[time]
            time_poi_dict[time] = list(filter(condition,current_poilist))
            print(time,":",time_poi_dict[time])

        f.close()

        return user_poi_dict,time_poi_dict,poi_geo_dict,valid_userlist,valid_poilist

def writeInfo(dict, file):
    with open(file, "w", encoding="utf-8") as f:
        for key in dict.keys():
            value = dict[key]
            f.write("{0}:{1}\n".format(key, value))
        f.close()

def write_valid_checkins(file_source,valid_userlist,valid_poilist,file_target):
    f1 = open(file_source,"r",encoding="utf-8")
    f2 = open(file_target,"w",encoding="utf-8")
    #选出有效的line，写入valid_data文件中
    for line in f1.readlines():
        info = line.split("	")
        uid = int(info[0])
        pid = int(info[5])
        if uid in valid_userlist and pid in valid_poilist:
            print("Write {0}".format(line))
            f2.write(line)

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
    f1 = open("../t_data/t_train_set.txt","w",encoding="utf-8")
    f2 = open("../t_data/t_test_set.txt","w",encoding="utf-8")
    f3 = open("../t_data/t_tune_set.txt","w",encoding="utf-8")
    #将控制台输出到文件
    f4 = open("../bugtest/twitter_promise_validation.txt","w",encoding="utf-8")
    for user in user_total.keys():
        #records是user的签到记录（str）
        records = user_total[user]
        poilist = []
        poi_record = {}
        for record in records:
            poi = int(record.split("	")[5])
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
        print("User {0}-------train set(size:{1},data:{2}".format(user,train_num,poilist1))
        f4.write("User {0}-------train set(size:{1},data:{2}\n".format(user,train_num,poilist1))

        condition1 = lambda c:c not in poilist1                   #筛选符合条件的列表元素

        filter_train_pois = list(filter(condition1,poilist))
        poilist2 = random.sample(filter_train_pois,test_num)

        for poi_id in poilist2:
            for line in poi_record[poi_id]:
                f2.write(line)
        print("User {0}-------test set(size:{1},data:{2}".format(user,test_num,poilist2))
        f4.write("User {0}-------test set(size:{1},data:{2}\n".format(user,test_num,poilist2))

        condition2 = lambda c:c not in poilist2

        poilist3 = list(filter(condition2,filter_train_pois))
        for poi_id in poilist3:
            for line in poi_record[poi_id]:
                f3.write(line)
        print("User {0}-------tune set(size:{1},data:{2}".format(user,tune_num,poilist3))
        f4.write("User {0}-------tune set(size:{1},data:{2}\n".format(user,tune_num,poilist3))

if __name__ == '__main__':
    user_poi_dict, time_poi_dict, poi_geo_dict, valid_userlist, valid_poilist = filter_user_poi(file_source)
    writeInfo(user_poi_dict, file_valid_user_visit)
    writeInfo(time_poi_dict, file_valid_visie_time)
    writeInfo(poi_geo_dict, file_valid_poi_list)
    write_valid_checkins(file_source, valid_userlist, valid_poilist, file_target)
    divide_train_test_tune(file_target)
