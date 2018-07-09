# 将测试集中的groundtruth以方便和推荐结果进行比较的形式组织起来
# Gowalla Twitter

import pickle

def readFile(file):
    userlist = []
    time_record = {}
    for t in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]:
        time_record[t] = []
    for line in file.readlines():
        info = line.split("	")
        # user_id = int(info[0])
        # time = int(info[4][11:13])
        user_id = int(info[0][5:])
        time = int(info[3][0:2])
        time_record[time].append(line)
        if user_id not in userlist:
            userlist.append(user_id)
            # print("加入测试集user：",user_id)
    return time_record,userlist

def format_file(time_record,userlist,file_groundtruth,file_usernum):
    user_num = {}
    user_poi = {}
    for record in time_record:
        print(record)
        info = record.split("	")
        # uid = int(info[0])
        # pid = int(info[5])
        uid = int(info[0][5:])
        pid = int(info[1][4:])
        if uid in user_poi.keys():
            if pid not in user_poi[uid]:
                user_poi[uid].append(pid)
                user_num[uid] += 1
        else:
            user_poi[uid] = [pid]
            user_num[uid] = 1
        # if uid in user_num.keys():
        #     user_num[uid] += 1
        # else:
        #     user_num[uid] = 1

    for user in userlist:
        if user in user_poi.keys():
            print("new groundtruth{0}:{1}".format(user,user_poi[user]))
            file_groundtruth.write("{0}:{1}\n".format(user,user_poi[user]))
    for user in userlist:
        if user in user_num.keys():
            print("new count{0}:{1}".format(user,user_num[user]))
            file_usernum.write("{0}:{1}\n".format(user,user_num[user]))

if __name__ == '__main__':
    # file_test = open("../t_data/t_test_set.txt","r",encoding="utf-8")
    # time_record,userlist = readFile(file_test)
    # file_test.close()
    # time_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    # for j in time_list:
        # file_groundtruth = open("../t_data/test/groundtruth/{0}positive.txt".format(j),"w",encoding="utf-8")
        # file_usernum = open("../t_data/test/groundtruth/{0}positive_usernum.txt".format(j),"w",encoding="utf-8")
        # format_file(time_record[j],userlist,file_groundtruth,file_usernum)
        # file_groundtruth.close()

    file_test = open("../f_data/f_test_set.txt","r",encoding="utf-8")
    time_record,userlist = readFile(file_test)
    file_test.close()
    time_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    for j in time_list:
        file_groundtruth = open("../f_data/test/groundtruth/{0}positive.txt".format(j),"w",encoding="utf-8")
        file_usernum = open("../f_data/test/groundtruth/{0}positive_usernum.txt".format(j),"w",encoding="utf-8")

        format_file(time_record[j],userlist,file_groundtruth,file_usernum)
        file_groundtruth.close()

    # file_test = open("../g_data/g_test_set.txt","r",encoding="utf-8")
    # time_record,userlist = readFile(file_test)
    # file_test.close()
    # time_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    # for j in time_list:
    #     file_groundtruth = open("../g_data/test/groundtruth/{0}positive.txt".format(j),"w",encoding="utf-8")
    #     file_usernum = open("../g_data/test/groundtruth/{0}positive_usernum.txt".format(j),"w",encoding="utf-8")
    #     format_file(time_record[j],userlist,file_groundtruth,file_usernum)
    #     file_groundtruth.close()

