# 为每个user创建其embedding向量
# 包含该user访问的poi的数学期望，与user的社交特征
import numpy as np

def readPOIS(file_pois):
    poi_rank_dict = {}
    with open(file_pois,"r",encoding="utf-8") as f:
        count = 0
        for line in f.readlines():
            poi_id = int(line.split(":")[0])
            poi_rank_dict[poi_id] = count
            count+=1
            print("FindNewPOI:{0},标号为{1}".format(poi_id,poi_rank_dict[poi_id]))
        f.close()
    return poi_rank_dict

######!!!!!!!!!注意一点，train_user_visit和total_user_visit中的user顺序是不同的!!!!!!!
def readUserVisit(file_train_user_visit,file_total_user_visit):
    user_visit_dict = {}
    userlist = []
    with open(file_train_user_visit,"r",encoding="utf-8") as f1:
        for line in f1.readlines():
            user_id = int(line.split(":")[0])
            pois_str = line.split(":")[1]
            pois = pois_str.replace("[","").replace("]\n","").split(", ")
            pois = list(map(int,pois))
            user_visit_dict[user_id] = pois
            print("User{0}访问列表：{1}".format(user_id,user_visit_dict[user_id]))
        f1.close()
    with open(file_total_user_visit,"r",encoding="utf-8") as f2:
        for line in f2.readlines():
            user_id = int(line.split(":")[0])
            userlist.append(user_id)
        print("User列表：",userlist)
        print(len(userlist))
        f2.close()
    return user_visit_dict,userlist

######!!!!!!!!!注意一点，train_user_visit和total_user_visit中的user顺序是不同的!!!!!!!
# 统一方法：这里是使用total_user_visit的顺序来排列user向量的
# poi的向量排列是不分训练集和总数据集的，它们的每个矩阵都总是严格一一对应的
def createUserEmbedding(user_visit_dict,poi_rank_dict,userlist,file_user_social,file_poi_embedding,file_out):
    user_social_matrix = np.loadtxt(open(file_total_user_social,"r",encoding="utf-8"),delimiter=",")
    poi_embedding_matrix = np.loadtxt(open(file_train_poi_embedding,"r",encoding="utf-8"),delimiter=",")
    user_average_pois = np.zeros((len(poi_embedding_matrix[:,0]),len(list(user_visit_dict.keys()))))

    # test代码
    # for i in range(10):
    #     print(i)
    #     print(poi_embedding_matrix[:,i])
    #     user_average_pois[:,0] += poi_embedding_matrix[:,i]
    # user_average_pois[:,0] /= 10
    # print(user_average_pois[:,0])
    # np.savetxt(file_out,user_average_pois,delimiter=",",fmt="%f")
    for user_rank in range(len(userlist)):           # range(len(userlist))
        user = userlist[user_rank]
        print("正在操作User{0}:{1}".format(user_rank,user))
        poi_rank_list = []
        poi_visited_list = user_visit_dict[user]
        for poi in poi_visited_list:
            poi_rank_list.append(poi_rank_dict[poi])
        print("User{0}的POI访问情况:{1}".format(user,poi_visited_list))
        print("访问的poi个数：",len(poi_visited_list))
        print("User{0}的POI对应标号:{1}".format(user,poi_rank_list))
        print("访问的poi个数：",len(poi_rank_list))
        print(user_average_pois[:,user_rank])
        for i in poi_rank_list:
            print(poi_embedding_matrix[:,i])
            user_average_pois[:,user_rank] += poi_embedding_matrix[:,i]
        user_average_pois[:,user_rank] /= len(poi_rank_list)
        print("User{0}的POI数学期望:{1}".format(user,user_average_pois[:user_rank]))

    total_feature_matrix = np.concatenate((user_average_pois,user_social_matrix))
    total_feature_matrix = np.insert(total_feature_matrix, 0, values=userlist, axis=0)
    print(total_feature_matrix)
    print("user向量维度：",len(total_feature_matrix[:,0]))
    print("user向量个数：",len(total_feature_matrix[0,:]))
    np.savetxt(file_out,total_feature_matrix,delimiter=",",fmt="%f")

if __name__ == '__main__':
    file_total_pois = "../data_5months/valid_total/g_valid_total_poi_geo.txt"
    file_train_user_visit = "../data_5months/train/g_train_user_visit.txt"
    file_total_user_visit = "../data_5months/valid_total/g_valid_total_user_visit.txt"
    poi_rank_dict = readPOIS(file_total_pois)
    user_visit_dict,userlist = readUserVisit(file_train_user_visit,file_total_user_visit)

    dimension_list = [20,40,60,80,100,120]
    for dim in dimension_list:
        print("======================Start {0} Features Task============================".format(dim))
        file_total_user_social = "../data_5months/valid_total_feature_matrixs/social/{0}user_social_feature.csv".format(dim)
        file_train_poi_embedding = "../data_5months/embeddings/poi/{0}poi_embedding.csv".format(dim)
        file_out = "../data_5months/embeddings/user/{0}user_embedding.csv".format(dim)
        createUserEmbedding(user_visit_dict,poi_rank_dict,userlist,file_total_user_social,file_train_poi_embedding,file_out)
    # test = np.zeros((64,1))
    # print(test)
    # test2 = np.zeros((1,64))
    # print(test2)
