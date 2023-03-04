import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model



df = pd.read_csv("F:\\Setups\\py-master\\py-master\\ML\\1_linear_reg\\Exercise\\canada_per_capita_income.csv")     #File location


plt.xlabel("per capita income US($)", fontsize = 10)		# naming the x plane
plt.ylabel("year", fontsize = 10)			                # naming the y plane
plt.scatter(df["per capita income US($)"], df.year, color= 'red', marker= '+')      #df["per capita income US($)"]
reg = linear_model.LinearRegression()
reg.fit(df[['year']], df["per capita income US($)"])
plt.plot(df['per capita income US($)'], df.year, color = 'blue')
a = reg.predict([[2020]])
b = reg.coef_
c = reg.intercept_
print(a)
print("m=",b,"y=",c)
plt.show()



# OR
# CHANGING
# x vs y
# plt.xlabel("year", fontsize = 10)                           # naming the x plane
# plt.ylabel("per capita income US($)", fontsize = 10)		# naming the y plane
# plt.scatter(df.year, df["per capita income US($)"],  color= 'red', marker= '+')      #df["per capita income US($)"]
# reg = linear_model.LinearRegression()
# reg.fit(df[['year']], df["per capita income US($)"])
# plt.plot(df.year , df['per capita income US($)'],  color = 'blue')
# a = reg.predict([[2020]])
# b = reg.coef_
# c = reg.intercept_
# print(a)
# print("m=",b,"y=",c)
# plt.show()
