# 测试numpy的argpartition函数

import numpy as np

result = np.random.randint(10,100,(10,1))
print(result)
print("=========================================")
# result = np.argpartition(result[:,0],5)[:5]
print("========================================")
result = np.argsort(-result[:,0])[:5]
print(result)
