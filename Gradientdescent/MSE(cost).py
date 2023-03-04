import numpy as np

def gradient_descent(x,y):
    b_curr: int
    m_curr = b_curr = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.006

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n)*sum([val**2 for val in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate*md
        b_curr = b_curr - learning_rate*bd
        print("m {}, b {}, cost {}, iterations {}". format(m_curr, b_curr, cost, i ))


x = np.array([2,4,6,8,10,12])
y = np.array([2,3,5,7,11,13])


gradient_descent(x,y)