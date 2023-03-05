import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

# '''
# Dummy variable using pandas
# '''
#
df = pd.read_csv("D:\\py-master\\ML\\5_one_hot_encoding\\homeprices.csv")
# dummies = pd.get_dummies(df.town)       # creating dummies attribute for town 3 column for 3 townships.
# merged = pd.concat([df,dummies], axis = 'columns')
# final = merged.drop(["town","west windsor"], axis = 'columns')
#
reg = linear_model.LinearRegression()
#
# X = final.drop("price", axis='columns')
# y = final.price
#
# reg.fit(X,y)
# # print(df)
# # print(final)
# '''
# one thing to note here is put 1 from the columns of township you wanna predict from.
# '''
#
# print(reg.predict([[2800,0,1]]))       # 2800 sqr ft home in robbinsville
# print(reg.predict([[3400,0,0]]))       # 3400 sqr ft home in west windsor
# print(reg.score(X,y))

'''
dummy variable using sklean
'''

le = preprocessing.LabelEncoder()

df.town = le.fit_transform(df.town)      # it takes the label column as input and convert them into dummy variable

X = df[['town','area']].values           # using .values because we want X to be a 2D array and not a data frame
y = df.price

# ohe = OneHotEncoder(categorical_features=[0])
ohe = ColumnTransformer([('one_hot_encoder', OneHotEncoder(),[0])], remainder='passthrough')
# X = ohe.fit_transform(X).toarray()
X = np.array(ohe.fit_transform(X), dtype = 'float')
X = X[:,1:]
reg.fit(X,y)

print(reg.predict([[1,0,2800]]))
print(reg.predict([[0,1,3400]]))