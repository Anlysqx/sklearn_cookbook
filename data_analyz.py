import pandas as pd
import matplotlib.pyplot as plt

Strat_test_set = pd.read_csv("data/Strat_test_set.csv")
Strat_train_set = pd.read_csv("data/Strat_train_set.csv")

# 创建一个副本以便可以随便尝试
housing = Strat_train_set.copy()
# 创建经纬度密度图
# housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1)
# 创建经纬度，人口，房价图
# housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,
#              s=housing['population']/100,label='population',
#              c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True)
# plt.savefig('longitude_latitude.png')
# plt.show()
# plt.legend()
# plt.savefig('population.png')
# plt.show()


# 下面查看每个属性和房屋中位数之间的相关系数
corr_matrix = housing.corr()
print(corr_matrix['median_house_value'].sort_values(ascending=False))

# 通过pandas的scatter_matrix 函数绘制相关系数图
# from pandas.tools.plotting import scatter_matrix
#
# attributes = ['median_house_value','median_income','total_rooms','housing_median_age']
# scatter_matrix(housing[attributes],figsize=(12,8))
# plt.savefig('corr.png')
# plt.show()

# 尝试特征组合
housing['rooms_Per_Household'] = housing['total_rooms']/housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
housing['population_per_household'] = housing['population']/housing['households']


corr_matrix = housing.corr()
print(corr_matrix['median_house_value'].sort_values(ascending=False))
