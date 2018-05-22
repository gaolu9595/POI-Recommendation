# 生成测试数据的类，DataGenerator

import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    '''为DNN生成测试数据，继承Sequence以实现多线程'''
    def __init__(self, user_emb, p_dict, time_emb, poilist):
        '''__init__方法会在类的对象被实例化时立即运行'''
        self.user_emb = user_emb
        self.p_dict = p_dict
        self.time_emb = time_emb
        self.poilist = poilist
        # self.__data_generation(self.user_emb, self.p_dict,self.time_emb, self.poilist)

    # 要实现len方法，才可以有steps = len(generator)
    # def __len__(self):
    #     size = len(self.poilist)
    #     length = int(size/self.batch_size)
    #     return length

    def __getitem__(self, user_emb, p_dict, time_emb, poilist):
        item = self.__data_generation(user_emb, p_dict, time_emb, poilist)
        return item

    def __data_generation(self, user_emb, p_dict, time_emb, poilist):
        '''Generates each batch data'''
        batch_size = 40
        batch_data = []
        # 开始格式化测试集
        j = 0
        step = 1
        for pid in poilist:
            value = np.concatenate((user_emb,p_dict[pid],time_emb),axis=0)
            if j < batch_size:
                batch_data.append(value)
                if j == 39 and step == 326:
                    yield np.array(batch_data)
                    break
                else:
                    j += 1
            else:
                yield np.array(batch_data)
                batch_data = [value]
                j = 1
                step += 1
