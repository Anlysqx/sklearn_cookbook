import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("data/housing.csv")

# 查看数据基本信息
# print(dataset.head())
# print(dataset.info())
# print(dataset['ocean_proximity'].value_counts())
# print(dataset.describe())

# 保存 查看数据分布图片
# dataset.hist(bins=50,figsize=(20,15))
# plt.savefig('img.png')
# plt.show()

#创建训练集和测试集
#如果希望每次的test集不变化，需要设置一个固定的random seed
def split_train_test(data,test_ratio):
    np.random.seed(55)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indicies = shuffled_indices[:test_set_size]
    train_indicies = shuffled_indices[test_set_size:]
    return data.iloc[train_indicies],data.iloc[test_set_size]

# 测试自己写的和sklearn自带的 train_test_split 函数功能一致
# train_data,test_data = train_test_split(dataset,test_size=0.2,random_state=55)
# train_data2,test_data2 = split_train_test(dataset,0.2)
# print(train_data)
# print(train_data2)

dataset['income_cat'] = np.ceil(dataset['median_income']/1.5)
dataset['income_cat'].where(dataset['income_cat'] < 5,5.0,inplace=True)

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=55)
for train_index,test_index in split.split(dataset,dataset['income_cat']):
    Strat_train_set = dataset.loc[train_index]
    Strat_test_set = dataset.loc[test_index]
# print(len(Strat_test_set)/len(dataset))
print("完全数据集上 income_cat 列范畴分布")
print(dataset['income_cat'].value_counts()/len(dataset['income_cat']))
print("分层抽样测试数据集上 income_cat 列范畴分布")
print(Strat_test_set['income_cat'].value_counts()/len(Strat_test_set))
train_data,test_data = train_test_split(dataset,test_size=0.2,random_state=55)
print("随机抽样测试数据集上 income_cat 列范畴分布")
print(test_data['income_cat'].value_counts()/len(test_data))

# 现在可以删除 income_cat 属性了
for set in (Strat_test_set,Strat_train_set):
    set.drop(["income_cat"],axis=1,inplace=True)

try:
    print(Strat_test_set['income_cat'])
except:
    print("删除成功")

pd.DataFrame.to_csv(Strat_train_set,'data/Strat_train_set.csv')
pd.DataFrame.to_csv(Strat_test_set,'data/Strat_test_set.csv')
assert os.path.exists('data/Strat_test_set.csv')