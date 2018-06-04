# 处理Foursquare数据集中的数据（California）
# 删除不合法的user和poi
# 划分训练集、调优集和测试集

file_source = "../foursquare/checkin_ca.txt"
file_target = "../foursquare/checkin_ca_valid.txt"
file_valid_user_visit = "../f_data/valid_total/f_valid_total_user_visit.txt"
file_valid_poi_list = "../f_data/valid_total/f_valid_total_poi_geo.txt"
file_valid_visie_time = "../f_data/valid_total/f_valid_total_visit_time.txt"

def filter_user_poi(file_total):
    with open(file_total,"r",encoding="utf-8") as f:
        user_poi_dict = {}
        time_poi_dict = {}
        poi_geo_dict = {}
        for line in f.readlines():
            info = line.split("	")
            user_id = int(info[0])
            time = info[1]
            # 这里的poi_id是字符串
            poi_id = info[2]
            geo_info = info[4].replace("{","").replace("}","").split(",")
            print(geo_info)
            geo = [float(geo_info[0]),float(geo_info[1])]
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

        print("====每个poi的访问时间信息====")
        for time in list(time_poi_dict.keys()):
            #filter时间字典中不合格的poi
            current_poilist = time_poi_dict[time]
            time_poi_dict[time] = list(filter(condition,current_poilist))
            print(time,":",time_poi_dict[time])

        f.close()
        return user_poi_dict,time_poi_dict,poi_geo_dict

def writeInfo(dict, file):
    with open(file, "w", encoding="utf-8") as f:
        for key in dict.keys():
            value = dict[key]
            f.write("{0}:{1}\n".format(key, value))
        f.close()

def write_valid_checkins(file_source,file_valid_user_visit,file_valid_poi_list,file_target):
    f1 = open(file_valid_user_visit,"r",encoding="utf-8")
    f2 = open(file_valid_poi_list,"r",encoding="utf-8")
    f3 = open(file_source,"r",encoding="utf-8")
    f4 = open(file_target,"w",encoding="utf-8")
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
    user_poi_dict, time_poi_dict, poi_geo_dict = filter_user_poi(file_source)
    writeInfo(user_poi_dict, file_valid_user_visit)
    writeInfo(time_poi_dict, file_valid_visie_time)
    writeInfo(poi_geo_dict, file_valid_poi_list)
    # write_valid_checkins(file_source,file_valid_user_visit,file_valid_poi_list,file_target)
