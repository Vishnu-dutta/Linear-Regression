import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model

df = pd.read_csv("F:\\Setups\\py-master\\py-master\\ML\\6_train_test_split\\carprices.csv")

X = df[["Mileage","Age(yrs)"]]
y = df["Sell Price($)"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=19)

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

print(reg.predict(X_test))
print(y_test)

print((reg.score(X_test, y_test)))
# print(reg.predict([[22500,2]]))

'''
random_state are just sets and finding the best set with maximum score or accuracy from this function here

def best_score(x,y):
    for i in range(30):
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=i)
        reg = linear_model.LinearRegression()
        reg.fit(X_train, y_train)
        a = (reg.score(X_test, y_test))
        print("score: {}, iteration: {}".format(a, i))


best_score(X,y)
'''

