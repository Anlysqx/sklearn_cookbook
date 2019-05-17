from sklearn.preprocessing import Imputer
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd



def get_train_and_test(dataset):
    dataset['median_income_cat'] = np.ceil(dataset['median_income'] / 1.5)
    dataset['median_income_cat'].where(dataset['median_income'] < 5, 5.0, inplace=True)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=55)
    for train_indices, test_indices in split.split(dataset, dataset['median_income_cat']):
        Strat_train = dataset.loc[train_indices]
        Strat_test = dataset.loc[test_indices]

    for set in (Strat_test, Strat_train):
        set.drop(['median_income_cat'], axis=1, inplace=True)
    return Strat_train,Strat_test

# 自定义转换器
from sklearn.base import BaseEstimator,TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
    def __init__(self,add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        rooms_per_household = X[:,rooms_ix]/X[:,household_ix]
        population_per_household = X[:,population_ix]/X[:,household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:,rooms_ix]/X[:,bedrooms_ix]
            return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]
        else:
            return np.c_[X,rooms_per_household,population_per_household]

dataset = pd.read_csv('data/housing.csv')
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
train_data,test_data = get_train_and_test(dataset)
print(type(train_data))
# train_data = pd.get_dummies(train_data,columns=['ocean_proximity'])
housing_num = train_data.drop(['ocean_proximity'],axis=1)


# from sklearn.preprocessing import LabelBinarizer
# encoder = LabelBinarizer()
# housing_cat = encoder.fit_transform(train_data.ocean_proximity)
# print(housing_cat)

class MyLableBinarize(BaseEstimator,TransformerMixin):
    def __init__(self,*args,**kwargs):
        self.encoder = LabelBinarizer(*args,**kwargs)
    def fit(self,x,y=0):
        self.encoder.fit(x)
        return self
    def transform(self,x,y=0):
        return self.encoder.transform(x)



from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn_features.transformers import DataFrameSelector
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
cat_attribs = ['ocean_proximity']
num_pipeline = Pipeline([
    ('selector',DataFrameSelector(list(housing_num))),
    ('imputer',Imputer(strategy="median")),
    ('attribs_adder',CombinedAttributesAdder()),
    ('std_scaler',StandardScaler()),
    ])

cat_pipeline = Pipeline([
    ('selector',DataFrameSelector(cat_attribs)),
    ('label_binarizer',MyLableBinarize()),
    ])
full_pipeline = FeatureUnion(transformer_list=[
    ('cat_pipeline',cat_pipeline),
    ('num_pipeline',num_pipeline),
])

housing_predict = full_pipeline.fit_transform(train_data)
print(len(housing_predict))
print(len(housing_predict[0]))
print(housing_predict)

# selector = DataFrameSelector(['ocean_proximity'])
#
# encoder = LabelBinarizer()
# ocean_cat = encoder.fit_transform(selector.fit_transform(train_data))
# print(len(ocean_cat))




