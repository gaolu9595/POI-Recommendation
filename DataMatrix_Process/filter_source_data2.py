# 筛选出gowalla数据集中美国的所有数据

file_source = "../gowalla/Gowalla_totalCheckins.txt"
file_target = "../gowalla/Gowalla_America_checkins.txt"
file_valid_user_visit = "../data/g_valid_total_user_visit.txt"
file_valid_poi_list = "../data/g_valid_total_poi_geo.txt"
file_valid_visie_time = "../data/g_valid_total_visit_time.txt"

def filterdata(file_source,file_target):
    with open(file_source,"r",encoding="utf-8") as f1:
        with open(file_target,"w",encoding="utf-8") as f2:
            count = 0
            for line in f1.readlines():
                info = line.split("	")
                user_id = int(info[0])
                # year = int(info[1][0:4])
                # month = int(info[1][5:7])
                geo = [float(info[2]),float(info[3])]
                poi_id = int(info[4])
                if int(geo[0]) in range(25,49) and int(geo[1]) in range(-130,-70):
                    f2.write(line)
                    count+=1
                    print("{0}:{1},{2}".format(count,user_id,poi_id))
            f2.close()
        f1.close()
        return f2

def filter_user_poi(file_total):
    with open(file_total,"r",encoding="utf-8") as f:
        user_poi_dict = {}
        time_poi_dict = {}
        poi_geo_dict = {}
        for line in f.readlines():
            info = line.split("	")
            user_id = int(info[0])
            time = info[1]
            geo = [float(info[2]),float(info[3])]
            poi_id = int(info[4])
            #构建用户访问poi列表的字典，只选取访问过美国本土poi的user
            #if int(geo[0]) in range(25,49) and int(geo[1]) in range(-130,-70):
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

        #过滤掉不活跃用户和poi，可以省好多好多时间啊！！
        #总是过滤不掉不合格的poi，是为什么啊？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
        #哇！！！！我真的蠢！过滤函数（lambda function)没有写对啊！之前过滤得到的是都不在有效poi列表中的poi...有毒啊有毒！
        print("====每个poi的地理位置经纬度信息====")
        for poi in list(poi_geo_dict.keys()):
            #filter签到用户数小于10的poi
            if len(poi_geo_dict[poi][1]) < 10:
                poi_geo_dict.pop(poi)
            else:
                print(poi,":",poi_geo_dict[poi])
                poi_geo_dict[poi] = poi_geo_dict[poi][0]
                print(poi,":",poi_geo_dict[poi])

        valid_poilist = list(poi_geo_dict.keys())            #有效的poilist
        print("有效POI列表：",valid_poilist)
        print("长度：",len(valid_poilist))
        condition = lambda c: c in valid_poilist          #过滤依靠的函数【哇！这个函数之前没写对，写成了not in，快把我坑死啦！】

        print("====每个用户访问过的poi列表====")
        for user in list(user_poi_dict.keys()):
            #filter签到poi少于15个的用户
            current_poilist = user_poi_dict[user]
            user_poi_dict[user] = list(filter(condition,current_poilist))
    #         for i in range(len(valid_poilist)):
				# for poi in user_poi_dict[user]:
    #             if poi not in list(poi_geo_dict.keys()):
    #                 user_poi_dict[user].remove(poi)      #pop是按位删除，remove是按值删除首个符合条件的元素
            if len(user_poi_dict[user]) < 15:
                # del user_poi_dict[user]
                user_poi_dict.pop(user)
            else:
                print(user,":",user_poi_dict[user])

        print("====每个poi的访问时间信息====")
        for time in list(time_poi_dict.keys()):
            #filter时间字典中不合格的poi
            current_poilist = time_poi_dict[time]
            time_poi_dict[time] = list(filter(condition,current_poilist))
            # visit_pois = time_poi_dict[time]
            # for poi in visit_pois:
            #     if poi not in list(poi_geo_dict.keys()):
            #         visit_pois.remove(poi)
            print(time,":",time_poi_dict[time])

        f.close()
        return user_poi_dict,time_poi_dict,poi_geo_dict

def writeInfo(dict, file):
    with open(file, "w", encoding="utf-8") as f:
        for key in dict.keys():
            value = dict[key]
            f.write("{0}:{1}\n".format(key, value))
        f.close()

def write_valid_checkins(file_total,file_valid_user_visit,file_valid_poi_list):
    f1 = open(file_valid_user_visit,"r",encoding="utf-8")
    f2 = open(file_valid_poi_list,"r",encoding="utf-8")
    f3 = open(file_total,"r",encoding="utf-8")
    f4 = open("../data/Gowalla_Valid_Data.txt","w",encoding="utf-8")
    userlist = []
    poilist = []
    # 构建有效的poi和user列表
    for line in f1.readlines():
        user = int(line.split(":")[0])
        if user not in userlist:
            userlist.append(user)
    print(userlist)
    for line in f2.readlines():
        poi = int(line.split(":")[0])
        if poi not in poilist:
            poilist.append(poi)
    print(poilist)
    #选出有效的line，写入valid_data文件中
    for line in f3.readlines():
        info = line.split("	")
        uid = int(info[0])
        pid = int(info[4])
        if uid in userlist and pid in poilist:
            print("Write {0}".format(line))
            f4.write(line)

if __name__ == '__main__':
    # filterdata(file_source,file_target)
    user_poi_dict, time_poi_dict, poi_geo_dict = filter_user_poi(file_target) #,poilist
    writeInfo(user_poi_dict, file_valid_user_visit)
    writeInfo(time_poi_dict, file_valid_visie_time)
    writeInfo(poi_geo_dict, file_valid_poi_list)
    write_valid_checkins(file_target,file_valid_user_visit,file_valid_poi_list)
