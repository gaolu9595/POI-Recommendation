
import numpy as np


file_valid_user = "../data_5months/valid_total/g_valid_total_user_visit.txt"
file_total_user_edges = "../gowalla/Gowalla_edges.txt"
file_total_user_social_matrix = "../data_5months/valid_total/g_total_user_social_matrix.txt"
file_valid_total_user_social = "../data_5months/valid_total/g_valid_total_user_social.txt"

#从总社交关系数据中统计出每一个用户的邻居关系（dict）
def findNeighbors(file_valid_user,file_total_user_edges):
    '''
    :param file: 总社交关系数据
    :return:邻居关系
    '''
    #
    f1 = open(file_valid_user,"r",encoding="utf-8")
    f2 = open(file_total_user_edges,"r",encoding="utf-8")
    print("========================FindValidUsers===========================")
    valid_user_list = []      #有效用户列表
    for line in f1.readlines():
        user_id = int(line.split(":")[0])
        if user_id not in valid_user_list:
            valid_user_list.append(user_id)
            print("Valid User:",user_id)
    print("========================ReadEdgesAndFilterUsers===========================")
    user_social_dict = {}
    for line in f2.readlines():
        edge = line.split("	")
        me_id,you_id = int(edge[0]),int(edge[1])
        #只找出有效用户的社交邻居关系
        if me_id in valid_user_list and you_id in valid_user_list:
            if me_id not in user_social_dict.keys():
                user_social_dict[me_id] = [you_id]
            else:
                user_social_dict[me_id].append(you_id)
            print("Find Neighbors：{0}------{1}".format(me_id,you_id))
    f1.close()
    f2.close()
    return user_social_dict, valid_user_list

def writeInfo(dict, file):
    with open(file, "w", encoding="utf-8") as f:
        for key in dict.keys():
            value = dict[key]
            f.write("{0}:{1}\n".format(key, value))
        f.close()

#构建用户社交关系矩阵
def create_user_social_matrix(user_social_dict, valid_userlist,file):
    '''
    :param final_social: dict
    :param user_list: list
    :return: matrix
    '''
    print("……开始初始化……")
    edge = len(valid_userlist)+1
    print(edge)
    matrix = np.zeros((edge, edge),dtype=np.int32)
    matrix[0][1:] = np.array(valid_userlist)
    i = 0
    j = 1
    while i<len(valid_userlist) and j<=edge:
        matrix[j][0] = valid_userlist[i]
        i += 1
        j += 1
    print("……首行首列赋值完成……")
    for row in range(1,len(matrix)):
        if row != 1:
            for col in range(1,row):
                 matrix[row][col] = matrix[col][row]
        # 若matrix[row][0]对应的用户有邻居，进行以下操作；否则，该行所有值均为0
        if matrix[row][0] in user_social_dict.keys():
            for col in range(row,len(matrix)):
                if matrix[0][col] in user_social_dict[matrix[row][0]]:
                    matrix[row][col] = 1
                else:
                    matrix[row][col] = 0
        else:
            for col in range(row,len(matrix)):
               matrix[row][col] = 0
            print("第{0}行对角线右侧值全为0".format(row))
    # 对数据进行行标准化处理
    for row in range(1,len(matrix)):
        data = matrix[row][1:]
        total_count = np.sum(data)
        for col in range(1,len(matrix)):
            if total_count != 0 :
                matrix[row][col] = matrix[row][col]/float(total_count)
            print("social位置[{0},{1}]:{2}".format(row, col, matrix[row][col]))
    # print("====user_social矩阵====")
    # for row in range(len(matrix)):
    #     print(matrix[row][:])
    print("========================WriteFile===========================")
    np.savetxt(file,matrix,delimiter=",",fmt="%f")
    print("========================Done!===========================")
    return matrix


if __name__ == '__main__':
    user_social_dict, valid_userlist = findNeighbors(file_valid_user,file_total_user_edges)
    # writeInfo(user_social_dict,file_valid_total_user_social)
    user_social_matirx = create_user_social_matrix(user_social_dict, valid_userlist, file_total_user_social_matrix)

