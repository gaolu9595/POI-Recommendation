# 失败的尝试

file_source = "../gowalla/Gowalla_totalCheckins.txt"
file_target = "../gowalla/Gowalla_5months_America.txt"
file_user_visit = "../data/g_valid_total_user_visit.txt"
file_valid_poi_list = "../data/g_valid_total_poi_geo.txt"

def filterdata(file_source,file_target):
    with open(file_source,"r",encoding="utf-8") as f1:
        with open(file_target,"w",encoding="utf-8") as f2:
            count = 0
            for line in f1.readlines():
                info = line.split("	")
                user_id = int(info[0])
                year = int(info[1][0:4])
                month = int(info[1][5:7])
                geo = [float(info[2]),float(info[3])]
                poi_id = int(info[4])
                if year==2010 and month in range(6,11):
                    if int(geo[0]) in range(25,49) and int(geo[1]) in range(-130,-70):
                        f2.write(line)
                        count+=1
                        print("{0}:{1},{2}".format(count,user_id,poi_id))
            f2.close()
        f1.close()
        return f2

def filter_user_poi(file_total,file_user_visit,file_valid_geo_list):
    f1 = open(file_total,"r",encoding="utf-8")
    f2 = open(file_user_visit,"r",encoding="utf-8")
    f3 = open(file_valid_poi_list,"r",encoding="utf-8")
    f4 = open("../gowalla/Gowalla_Valid_Data.txt","w",encoding="utf-8")
    f5 = open("../bugtest/promise_validation.txt","w",encoding="utf-8")
    #获取有效的poi列表
    valid_poilist = []
    for line in f3.readlines():
        poi = int(line.split(":")[0])
        valid_poilist.append(poi)
    print(valid_poilist)
    f5.write("获取到的有效poi个数（应为15524）：{0}".format(len(valid_poilist)))
    #获取有效的user-poi对
    user_poilist_dict = {}
    for line in f2.readlines():
        line = line.replace("[","").replace("]\n","")
        info = line.split(":")
        user = int(info[0])
        pois = info[1].split(", ")
        for i in range(len(pois)):
            pois[i] = int(pois[i])
        f5.write("该用户本来访问的poi数目:{0}".format(len(pois)))
        condition = lambda c:c not in valid_poilist
        poilist = list(filter(condition,pois))
        print("{0}:{1}".format(user,poilist))
        f5.write("该用户访问的有效poi数目（应该小于等于上面的数字）:{0}".format(len(poilist)))
        user_poilist_dict[user] = poilist
    #只选取在valid_user_visit中存在的user_poi对，放入总数据中
    validation_pois = []
    for line in f1.readlines():
        info = line.split("	")
        user = int(info[0])
        print(user)
        poi = int(info[4])
        print(poi)
        #验证有效集中到底有多少poi
        if poi not in validation_pois:
            validation_pois.append(poi)
        if user in user_poilist_dict.keys():
            if len(user_poilist_dict[user]) > 15:
                if poi in user_poilist_dict[user]:
                    print("write:{0}".format(line))
                    f4.write(line)
    f5.write("有效集中的总poi数目：{0}".format(len(validation_pois)))


if __name__ == '__main__':
    #filterdata(file_source,file_target)
    filter_user_poi(file_target,file_user_visit,file_valid_poi_list)
