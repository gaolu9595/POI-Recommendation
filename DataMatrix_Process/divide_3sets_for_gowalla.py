# 处理Gowalla（2013UTE）有效数据集中的数据

file_source = "../g_data/checkins_valid.txt"
file_valid_user_visit = "../g_data/valid_total/g_valid_total_user_visit.txt"
file_valid_poi_list = "../g_data/valid_total/g_valid_total_poi_geo.txt"
file_valid_visie_time = "../g_data/valid_total/g_valid_total_visit_time.txt"

def filter_user_poi(file_total):
    with open(file_total,"r",encoding="utf-8") as f:
        user_poi_dict = {}
        time_poi_dict = {}
        poi_geo_dict = {}
        for line in f.readlines():
            info = line.split("	")
            user_id = int(info[0][5:])
            time = info[3]
            poi_id = int(info[1][4:])
            geo_info = info[2].split(",")
            geo = [float(geo_info[0]),float(geo_info[1])]
            #构建用户访问poi列表的字典
            if user_id in user_poi_dict.keys():
                if poi_id not in user_poi_dict[user_id]:      #保证一个user访问过的每个poi只出现一次
                    user_poi_dict[user_id].append(poi_id)
            else:
                user_poi_dict[user_id] = [poi_id]
                print("User",user_id)
            #对时间格式进行预处理,获取签到行为的时间点(24小时制)
            time = int(time[0:2])
             #构建poi被访问时间的字典，只选取部分时间内的poi签到记录
            if time in time_poi_dict.keys():
                #每个时间的poi列表是有重复元素的
                time_poi_dict[time].append(poi_id)          #一个时间内的poi可以出现多次
            else:
                time_poi_dict[time] = [poi_id]
            #构建poi地理位置字典
            if poi_id not in poi_geo_dict.keys():
                poi_geo_dict[poi_id] = geo

        f.close()
        return user_poi_dict,time_poi_dict,poi_geo_dict

def writeInfo(dict, file):
    with open(file, "w", encoding="utf-8") as f:
        for key in dict.keys():
            value = dict[key]
            f.write("{0}:{1}\n".format(key, value))
        f.close()

if __name__ == '__main__':
    user_poi_dict, time_poi_dict, poi_geo_dict = filter_user_poi(file_source)
    writeInfo(user_poi_dict, file_valid_user_visit)
    writeInfo(time_poi_dict, file_valid_visie_time)
    writeInfo(poi_geo_dict, file_valid_poi_list)
