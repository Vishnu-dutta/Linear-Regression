import pandas as pd
import math
from word2number import w2n
from sklearn import linear_model

df = pd.read_csv("F:\\Setups\\py-master\\py-master\\ML\\2_linear_reg_multivariate\\Exercise\\hiring.csv")

# took mean and filled the empty scores
mean_score = math.floor(df["test_score(out of 10)"].mean())
df["test_score(out of 10)"] = df["test_score(out of 10)"].fillna(mean_score)

# filling the emppty spaces with zero here
df["experience"] = df["experience"].fillna('zero')

# changing the words to numbers to further apply LR (to note .apply())
df["experience"] = df["experience"].apply(w2n.word_to_num)

reg = linear_model.LinearRegression()
reg.fit(df[["experience", "test_score(out of 10)", "interview_score(out of 10)"]], df["salary($)"])


print("m: {}, b: {}".format(reg.coef_, reg.intercept_))

print("salary1($): {}".format(reg.predict([[2, 9, 6]])))
print("salary2($): {}".format(reg.predict([[12, 10, 10]])))


