import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import numpy as np


df = pd.read_csv("F:\\Setups\\py-master\\py-master\\ML\\5_one_hot_encoding\\Exercise\\carprices.csv")

dummies = pd.get_dummies(df["Car Model"])
merged = pd.concat([df, dummies], axis="columns")
final = merged.drop(["Car Model", "BMW X5"], axis="columns")

reg = linear_model.LinearRegression()
X = final.drop(["Sell Price($)"], axis="columns")
y = final["Sell Price($)"]

reg.fit(X,y)

print(final)
print(reg.predict([[45000,4,0,1]]))                                 # mercedez benz that is 4 yr old with mileage 45000
print(reg.predict([[86000,7,0,0]]))                                 # BMW X5 that is 7 yr old with mileage 86000
print("complete dataset score: {}".format(reg.score(X,y)))          # score(accuracy of the model)

# training and testing dataset

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=25)
print(y_test)
print(reg.predict(X_test))
print("trained dataset score: {}".format(reg.score(X_test, y_test)))

'''
def best_score(x,y):
    for i in range(30):
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=i)
        reg = linear_model.LinearRegression()
        reg.fit(X_train, y_train)
        a = (reg.score(X_test, y_test))
        print("score: {}, iteration: {}".format(a,i))

best_score(X,y)

'''