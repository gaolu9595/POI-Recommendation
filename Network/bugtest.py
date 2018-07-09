# 构建DNN 【Train Task: Binary Classification】

import numpy as np
from numpy import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras import regularizers,callbacks,optimizers

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


if __name__ == '__main__':
    print("=============================BuildModel===============================")
    epochs = 100     # 迭代次数【感觉迭代次数尽可能多点吧，但过多可能就过拟合了？】
    batch_size = 64      # 批次样本大小# 创建DNN模型
    # sigmoid对应binary_crossentropy二分类
    # softmax对应categorical_crossentropy多分类
    # BN层加在Relu函数的后面，貌似会有效防止过拟合，虽然训练时准确率没那么高，但是预测时效果好一点
    model = Sequential()
    # model.add(BatchNormalization(input_shape=(1272,)))

    model.add(Dense(2048, activation="relu", kernel_initializer='glorot_normal', bias_initializer='zeros', input_dim=1272))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # model.add(BatchNormalization())
    model.add(Dense(1024, activation="relu", kernel_initializer='glorot_normal', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # model.add(BatchNormalization())
    model.add(Dense(512, activation="relu", kernel_initializer='glorot_normal', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # model.add(BatchNormalization())
    model.add(Dense(256, activation="relu", kernel_initializer='glorot_normal', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # model.add(BatchNormalization())
    model.add(Dense(128, activation="relu", kernel_initializer='glorot_normal', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # model.add(BatchNormalization())
    model.add(Dense(64, activation="relu", kernel_initializer='glorot_normal', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # model.add(BatchNormalization())
    model.add(Dense(32, activation="relu", kernel_initializer='glorot_normal', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # model.add(BatchNormalization())
    model.add(Dense(1, activation="sigmoid", kernel_initializer='glorot_normal', bias_initializer='zeros'))
    # model.add(BatchNormalization())
    # 打印网络结构和参数信息
    model.summary()
    # 将模型用图画出来,Graphviz和pydot-ng是可视化工具
    # plot_model(model,to_file="dnn_model.png")
    # compile编译网络模型：指定损失函数、优化算法和评价标准等
    # RecNet中使用的是Adam优化器，默认参数遵循Adam原论文中提供的值
    # sgd = optimizers.SGD(lr=0.1, momentum=0.9, decay=0.005)
    model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

    # after normalization
    print("=============================StartReadingData===============================")
    file_train = "../f_data/train_format_input/neighbor/f_train_format_300embedding.csv"
    file_tune = "../f_data/tune_format_input/neighbor/f_tune_format_300embedding.csv"
    x_train, y_train = loadTrainData(file_train)
    x_tune, y_tune = loadTuneData(file_tune)
    print("=============================StartTraining=====================================")
    early_stop = callbacks.EarlyStopping(monitor="loss", min_delta=0, patience=10, mode="auto", verbose=1)
    checkpoint = callbacks.ModelCheckpoint("./dnn_version/f_neighbor/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=False, mode="auto", period=1)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_tune, y_tune), shuffle=True, verbose=1, callbacks=[early_stop, checkpoint])
    # model.save('./dnn_version/f_neighbor/300dim_normal_64_100epoch.h5')
    print("=============================StartTuning=====================================")
    score = model.evaluate(x_tune, y_tune, batch_size=batch_size, verbose=1)
    print(model.metrics_names)
    print(score)

