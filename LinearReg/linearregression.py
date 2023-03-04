import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model



df = pd.read_csv("F:\\homeprices.csv")


plt.xlabel("area(square ft)", fontsize = 20)		# naming the x plane
plt.ylabel("price(US$)", fontsize = 20)			    # naming the y plane
plt.scatter(df.area, df.price, color= 'red', marker= '+')
reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)
plt.plot(df.area, reg.predict(df[['area']]), color = 'blue')
a = reg.predict([[3300]])
b = reg.coef_
c = reg.intercept_
print(a)
print("m=",b,"y=",c)
plt.show()


