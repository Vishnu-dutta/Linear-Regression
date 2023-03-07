import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model
import numpy as np
from sklearn import preprocessing

df = pd.read_csv("F:\\Setups\\py-master\\py-master\\ML\\5_one_hot_encoding\\Exercise\\carprices.csv")
le = preprocessing.LabelEncoder()
reg = linear_model.LinearRegression()
df["Car Model"] = le.fit_transform(df["Car Model"])
X = df[["Car Model", "Mileage", "Age(yrs)"]].values

ohe = ColumnTransformer([("one_hot_encoding", OneHotEncoder(),[0])], remainder="passthrough")
X = np.array(ohe.fit_transform(X), dtype = "float")
X = X[:,1:]
y = df["Sell Price($)"]

reg.fit(X,y)
print(df)
print(X)
print(reg.predict([[0, 1, 45000, 4]]))          # Mercedez benz that is 4 yr old with mileage 45000
print(reg.predict([[1, 0, 86000, 7]]))          # BMW X5 that is 7 yr old with mileage 86000
print(reg.score(X,y))                           # score(accuracy of the model)

