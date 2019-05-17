import pandas as pd
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn_features.transformers import DataFrameSelector
from sklearn.preprocessing import LabelBinarizer, Imputer
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('data/housing.csv')
from sklearn.model_selection import StratifiedShuffleSplit

dataset['median_income_cat'] = np.ceil(dataset['median_income'] / 1.5)
dataset['median_income_cat'].where(dataset['median_income'] < 5, 5.0, inplace=True)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=55)
for train_indices, test_indices in split.split(dataset, dataset['median_income_cat']):
    Strat_train = dataset.loc[train_indices]
    Strat_test = dataset.loc[test_indices]

for set in (Strat_test, Strat_train):
    set.drop(['median_income_cat'], axis=1, inplace=True)

train_data_label = Strat_train['median_house_value']
train_data = Strat_train.drop(['median_house_value'],axis=1)
test_data = Strat_test


housing_num = train_data.drop(['ocean_proximity'],axis=1)
from sklearn.base import BaseEstimator,TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, rooms_ix] / X[:, bedrooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


class MyLableBinarize(BaseEstimator, TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)

    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self

    def transform(self, x, y=0):
        return self.encoder.transform(x)


cat_attribs = ['ocean_proximity']
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(list(housing_num))),
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', MyLableBinarize()),
])
full_pipeline = FeatureUnion(transformer_list=[
    ('cat_pipeline', cat_pipeline),
    ('num_pipeline', num_pipeline),
])

housing_predict = full_pipeline.fit_transform(train_data)
# print(housing_predict)

# 先训练一个线性模型
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(housing_predict,train_data_label)

test_data_label = test_data['median_house_value']
test_data = test_data.drop(['median_house_value'],axis=1)
test_data = full_pipeline.transform(test_data)
test_predict = linear_reg.predict(test_data)
train_predict = linear_reg.predict(housing_predict)

from sklearn.metrics import mean_squared_error
err_train = mean_squared_error(train_data_label,train_predict)
err_test = mean_squared_error(test_data_label,test_predict)
print("test_value_max = ",np.max(test_data_label),"test_value_min = ",np.min(test_data_label))
print("linear train err = ",np.sqrt(err_train))
print("linear test err = ",np.sqrt(err_test))

from sklearn.tree import DecisionTreeRegressor
treeReg = DecisionTreeRegressor()
treeReg.fit(housing_predict,train_data_label)
train_predict = treeReg.predict(housing_predict)
test_predict = treeReg.predict(test_data)
err_train = mean_squared_error(train_data_label,train_predict)
err_test= mean_squared_error(test_data_label,test_predict)
print("tree train err = ",np.sqrt(err_train))
print("tree test err = ",np.sqrt(err_test))

# 十折交叉验证
# from sklearn.model_selection import cross_val_score
# scroes = cross_val_score(treeReg,housing_predict,train_data_label,scoring="neg_mean_squared_error",cv=10)
# print(np.sqrt(-scroes).mean())
# print(np.sqrt(-scroes).std())
#
# scroes = cross_val_score(linear_reg,housing_predict,train_data_label,scoring="neg_mean_squared_error",cv=10)
# print(np.sqrt(-scroes).mean())
# print(np.sqrt(-scroes).std())

from sklearn.ensemble import RandomForestRegressor
randomfReg = RandomForestRegressor()
randomfReg.fit(housing_predict,train_data_label)
train_predict = randomfReg.predict(housing_predict)
err_train = mean_squared_error(train_predict,train_data_label)
err_train = np.sqrt(err_train)
print("err_train = ",err_train)
# print("下面进行十折交叉验证")
# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(randomfReg,housing_predict,train_data_label,scoring="neg_mean_squared_error",cv=10)
# print(np.sqrt(-scores).mean())
# print(np.sqrt(-scores).std())

# from sklearn.externals import joblib
#
# myForestModel = joblib.dump(randomfReg,'randomfReg.pkl')
# new_myForestModel = joblib.load('randomfReg.pkl')
# print(new_myForestModel.predict(test_data))

# 用网格搜索来微调模型

# from sklearn.model_selection import GridSearchCV
# param_grid = [
#     {'n_estimators':[30,40,50],'max_features':[6,7]},
#     {'bootstrap':[False],'n_estimators':[30,40],'max_features':[6,7,8]}
# ]
# fore_reg = RandomForestRegressor()
# grid_search = GridSearchCV(fore_reg,param_grid=param_grid,cv=5,scoring="neg_mean_squared_error")
# grid_search.fit(housing_predict,train_data_label)
# print(grid_search.best_params_)

fore_reg = RandomForestRegressor(max_features=6,n_estimators=40)
fore_reg.fit(housing_predict,train_data_label)
train_predict = fore_reg.predict(housing_predict)
err_train = mean_squared_error(train_predict,train_data_label)
print("grid_err_train = ",np.sqrt(err_train))

# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(fore_reg,housing_predict,train_data_label,scoring="neg_mean_squared_error",cv=10)
# print("10 fold cross error.mean = ",np.sqrt(-scores).mean(),",error.std = ",np.sqrt(-scores).std())

from sklearn import svm
svr = svm.SVR(kernel="linear")
svr.fit(housing_predict,train_data_label)
train_predict = svr.predict(housing_predict)
err_train = mean_squared_error(train_predict,train_data_label)
print("svr err_train = ",np.sqrt(err_train))
