# 将测试集中的groundtruth以方便和推荐结果进行比较的形式组织起来
# notime

import pickle

def format_file(file,file_groundtruth,file_usernum):
    userlist = []
    user_num = {}
    user_poi = {}
    for line in file.readlines():
        info = line.split("	")
        uid = int(info[0])
        pid = int(info[5])
        if uid not in userlist:
            userlist.append(uid)
        
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
    file_test = open("../t_data/t_test_set.txt","r",encoding="utf-8")
    # userlist = readFile(file_test)
    file_groundtruth = open("../t_data/test/groundtruth/notime/positive.txt","w",encoding="utf-8")
    file_usernum = open("../t_data/test/groundtruth/notime/positive_usernum.txt","w",encoding="utf-8")
    
    format_file(file_test,file_groundtruth,file_usernum)
    
    file_test.close()
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
