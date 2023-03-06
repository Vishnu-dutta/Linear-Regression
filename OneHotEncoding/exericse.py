import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
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
print(reg.predict([[45000,0,1,4]]))     # mercedez benz that is 4 yr old with mileage 45000
print(reg.predict([[86000,0,0,7]]))     # BMW X5 that is 7 yr old with mileage 86000
print(reg.score(X,y))                   # score(accuracy of the model)
