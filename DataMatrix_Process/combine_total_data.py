# 用Quan Yuan在2013年的数据集，合并训练集测试集和调优集
def read3sets():
    f1 = open("../gowalla/train.txt","r",encoding="utf-8")
    f2 = open("../gowalla/test.txt","r",encoding="utf-8")
    f3 = open("../gowalla/tune.txt","r",encoding="utf-8")
    file_total = open("../gowalla/QuanYuan_total_checkins.txt","w",encoding="utf-8")

    for line in f1.readlines():
        file_total.write(line)
    for line in f2.readlines():
        file_total.write(line)
    for line in f3.readlines():
        file_total.write(line)
    file_total.close()
    f3.close()
    f2.close()
    f1.close()

def read_user_poi_info(file_total):
    with open(file_total,"r",encoding="utf-8") as f:
        user_poi_dict = {}
        time_poi_dict = {}
        poi_geo_dict = {}
        for line in f.readlines():
            info = line.split("	")
            user_id = info[0]
            time = info[3]
            # print(info[2])
            lat = float(info[2].split(",")[0])
            # print(lat)
            lon = float(info[2].split(",")[1])
            # print(lon)
            geo = [lat, lon]
            # print(geo)
            poi_id = info[1]
            # 构建用户访问poi的信息列表
            if user_id in user_poi_dict.keys():
                if poi_id not in user_poi_dict[user_id]:      # 保证一个user访问过的每个poi只出现一次
                    user_poi_dict[user_id].append(poi_id)
            else:
                user_poi_dict[user_id] = [poi_id]
                print("User", user_id)
            # 构建poi被访问时间的字典，只选取部分时间内的poi签到记录
            time = int(time[0:2])
            if time in time_poi_dict.keys():
                # 每个时间的poi列表是有重复元素的
                time_poi_dict[time].append(poi_id)          # 一个时间内的poi可以出现多次
            else:
                time_poi_dict[time] = [poi_id]
                print("Time", time)
            # 构建poi的地理信息列表
            if poi_id not in poi_geo_dict.keys():
                poi_geo_dict[poi_id] = geo
                print("POI", poi_id)
        f.close()
    return user_poi_dict, time_poi_dict, poi_geo_dict

def writeInfo(dict, file):
    with open(file, "w", encoding="utf-8") as f:
        for key in dict.keys():
            value = dict[key]
            f.write("{0}:{1}\n".format(key, value))
        f.close()

if __name__ == '__main__':
    # read3sets()
    file_total = "../gowalla/QuanYuan_total_checkins.txt"
    file_valid_user_visit = "../g_data/total_user_visit.txt"
    file_valid_poi_list = "../g_data/total_poi_geo.txt"
    file_valid_visit_time = "../g_data/total_visit_time.txt"
    # read_user_poi_info(file_total)
    user_poi_dict, time_poi_dict, poi_geo_dict = read_user_poi_info(file_total)
    writeInfo(user_poi_dict, file_valid_user_visit)
    writeInfo(time_poi_dict, file_valid_visit_time)
    writeInfo(poi_geo_dict, file_valid_poi_list)
