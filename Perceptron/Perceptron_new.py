# -*- coding:utf-8 -*-
# Filename: train2.2.py
# Author：hankcs
# Date: 2015/1/31 15:15
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

# An example in that book, the training set and parameters' sizes are fixed
training_set = np.array([[[3, 3], 1], [[4, 3], 1], [[1, 1], -1]])

a = np.zeros(len(training_set), np.float)
b = 0.0
Gram = None
y = np.array(training_set[:, 1])
x = np.empty((len(training_set), 2), np.float)
for i in range(len(training_set)):
    x[i] = training_set[i][0]
history = []
w = None


def cal_gram():
    """
    calculate the Gram matrix
    :return:
    """
    g = np.empty((len(training_set), len(training_set)), np.int)
    for i in range(len(training_set)):
        for j in range(len(training_set)):
            g[i][j] = np.dot(training_set[i][0], training_set[j][0])
    return g


def update(i):
    """
    update parameters using stochastic gradient descent
    :param i:
    :return:
    """
    global a, b
    a[i] += 1
    b = b + y[i]
    history.append([np.dot(a * y, x), b])
    # print a, b # you can uncomment this line to check the process of stochastic gradient descent


# calculate the judge condition
def cal(i):
    global a, b, x, y

    res = np.dot(a * y, Gram[i])
    res = (res + b) * y[i]
    return res


# check if the hyperplane can classify the examples correctly
def check():
    global a, b, x, y
    flag = False
    for i in range(len(training_set)):
        if cal(i) <= 0:
            flag = True
            update(i)
    if not flag:
        global w
        w = np.dot(a * y, x)
        print("RESULT: w: " + str(w) + " b: " + str(b))
        return False
    return True

def pre(x):
    global w,b
    print(f(np.dot(w,x)+ b))

def f(x):
    '''
    感知器使用的激活函数就是一个sign符号函数
    :param x:
    :return:
    '''
    return 1 if x > 0 else -1

def plotData(X,y,b,theta):
    plt.xlabel('x1')
    plt.ylabel('x2')
    m = len(y)
    for i in range(m):
        if int(y[i]) == 1:
            plt.scatter(X[i][0], X[i][1], marker='o', color='red')
        else:
            plt.scatter(X[i][0], X[i][1], marker='x', color='blue')
    print('theta', theta[0], theta[1])
    xl = np.arange(0, 10, 0.001)
    yl = -1 / theta[1] * (b + theta[0] * xl)
    plt.plot(xl, yl, color='black', linewidth='1')
    plt.show()

if __name__ == "__main__":
    Gram = cal_gram()  # initialize the Gram matrix
    for i in range(1000):
        if not check(): break
    print(w,b)
    plotData(x,y,b,w)
    pre(np.array([3, 1]))