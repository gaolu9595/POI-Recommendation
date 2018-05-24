# 构建DNN 【Train Task: Binary Classification】
# user向量和poi向量分别为84、64维

import numpy as np
from numpy import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras import regularizers,callbacks

# 获取训练集
def loadTrainData(file):
    data = np.loadtxt(open(file,"r",encoding="utf-8"),delimiter=",")
    x_data = []
    y_data = []
    dim = int(len(data[0,:]))
    for i in range(int(dim/2)):
        x_data.append(data[:,i])
        if i == dim/2-1:
            print("positive sample{0}:{1}".format(i,data[:,i]))
        y_data.append(1)
    for j in range(int(dim/2),dim):
        x_data.append(data[:,j])
        if j == dim-1:
            print("negative sample{0}:{1}".format(j,data[:,j]))
        y_data.append(0)
    return array(x_data), array(y_data)

# 获取调优集
def loadTuneData(file):
    tune = np.loadtxt(open(file,"r",encoding="utf-8"),delimiter=",")
    x_tune = []
    y_tune = []
    for i in range(len(tune[0,:])):
        x_tune.append(tune[:,i])
        if i==len(tune[0,:])-1:
            print("last tune sample{0}:{1}".format(i,tune[:,i]))
        y_tune.append(1)
    return array(x_tune), array(y_tune)
# 获取测试集   测试集是含有22亿个条目的稀疏矩阵
# def loadTestData(file):
#     test = np.loadtxt(open(file,"r",encoding="utf-8"),delimiter=",")
#     x_test = []
#     for i in range(len(test[0,:])):
#         x_test.append(test[:,i])
#     return array(x_test)

# 自定义带正则项的损失函数
# def loss_func_create():


if __name__ == '__main__':
    print("=============================BuildModel===============================")
    epochs = 20     # 迭代次数【感觉迭代次数尽可能多点吧，但过多可能就过拟合了？】
    batch_size = 40      # 批次样本大小# 创建DNN模型
    # sigmoid对应binary_crossentropy二分类
    # softmax对应categorical_crossentropy多分类
    # BN层加在Relu函数的后面，貌似会有效防止过拟合，虽然训练时准确率没那么高，但是预测时效果好一点
    model = Sequential()
    # model.add(BatchNormalization(input_shape=(172,)))
    model.add(Dense(1024, activation="relu", kernel_initializer='random_uniform', input_dim=172, kernel_regularizer=regularizers.l2(0.)))
    model.add(BatchNormalization(input_shape=(1024,)))
    model.add(Dropout(0.5))

    # model.add(BatchNormalization())
    model.add(Dense(512, activation="relu", kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # model.add(BatchNormalization())
    model.add(Dense(256, activation="relu", kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(BatchNormalization())
    model.add(Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.)))
    # model.add(BatchNormalization())
    # 打印网络结构和参数信息
    model.summary()
    # 将模型用图画出来,Graphviz和pydot-ng是可视化工具
    # plot_model(model,to_file="dnn_model.png")
    # compile编译网络模型：指定损失函数、优化算法和评价标准等       # RecNet中使用的是Adam优化器，默认参数遵循Adam原论文中提供的值
    # loss_function = loss_func_create()
    model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

    print("=============================StartReadingData===============================")
    file_train = "../data_5months/train_format_input/g_train_format_20embedding.csv"
    file_tune = "../data_5months/tune_format_input/g_tune_format_20embedding.csv"
    x_train, y_train = loadTrainData(file_train)
    x_tune, y_tune = loadTuneData(file_tune)
    print("=============================StartTraining=====================================")
    early_stop = callbacks.EarlyStopping(monitor="acc", min_delta=0, patience=10, mode="auto", verbose=1)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[early_stop])
    model.save('dnn_trained_20dim_4km_ver1.h5')
    print(model.get_weights())
    print("=============================StartTuning=====================================")
    score = model.evaluate(x_tune, y_tune, batch_size=100, verbose=1)
    print(model.metrics_names)
    print(score)

