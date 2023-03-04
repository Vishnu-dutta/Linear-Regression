import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


def gradient_descent(x,y):
    m_curr = 0
    b_curr = 0
    n = len(x)
    learning_rate = 0.0002
    iterations = 1000000

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr

        cost = (1/n)*sum([val**2 for val in (y - y_predicted)])

        md = -(2/n)*sum(x*(y - y_predicted))
        bd = -(2/n)*sum(y - y_predicted)

        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd

    return m_curr, b_curr

def sklearning():
    df = pd.read_csv("F:\\Setups\\py-master\\py-master\\ML\\3_gradient_descent\\Exercise\\test_scores.csv")
    reg = linear_model.LinearRegression()
    reg.fit(df[["math"]], df.cs)
    return reg.coef_, reg.intercept_





df = pd.read_csv("F:\\Setups\\py-master\\py-master\\ML\\3_gradient_descent\\Exercise\\test_scores.csv")
plt.scatter(df.math, df.cs)
x = np.array(df.math)
y = np.array(df.cs)

m, b = gradient_descent(x,y)
ms, bs =sklearning()

print("using gradient descent function: coef: {}, intercept: {} ".format(m,b))

print("using sklearn lib : coef: {}, intercept: {} ".format(ms,bs))




