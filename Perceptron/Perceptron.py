import numpy as np
from matplotlib import pyplot as plt


class Perceptron(object):
    def __init__(self, input_num, activator):
        '''

        :param input_num: 感知器数据的输入维度
        :param activator: 激活函数
        '''
        self.input_num = input_num
        self.activator = activator
        # 权重初始化
        self.weights = [0.0 for _ in range(input_num)]
        # 偏置项初始化
        self.bias = 0.0


alph = 0.01


def costFunction(X, y, theta, b):
    return y * (X.dot(theta) + b)


def train(X, y, theta, b):
    while True:
        # i = random.randint(0,len(y)-1)
        for i in range(len(y)):
            result = y[i][0] * ((np.dot(X[i:i + 1], theta)) + b)
            if result <= 0:
                temp = np.reshape(X[i:i + 1], (theta.shape[0], 1))
                theta += y[i][0] * temp * alph
                b += y[i][0] * alph
        cost = costFunction(X, y, theta, b)
        # print(cost)
        if (cost > 0).all():
            break
    return theta, b


def plotData(X, y, b, theta):
    plt.xlabel('x1')
    plt.ylabel('x2')
    m = len(y)
    for i in range(m):
        if int(y[i][0]) == 1:
            plt.scatter(X[i][0], X[i][1], marker='x', color='red')
        else:
            plt.scatter(X[i][0], X[i][1], marker='x', color='blue')
    print('theta', theta[0][0], theta[1][0])
    xl = np.arange(0, 10, 0.001)
    yl = -1 / theta[1][0] * (b + theta[0][0] * xl)
    plt.plot(xl, yl, color='black', linewidth='1')
    plt.show()


def plotData2(X, y):
    plt.xlabel('x1')
    plt.ylabel('x2')
    m = len(y)
    for i in range(m):
        if int(y[i][0]) == 1:
            plt.scatter(X[i][0], X[i][1], marker='x', color='red')
        else:
            plt.scatter(X[i][0], X[i][1], marker='x', color='blue')
    plt.show()


def f(x):
    '''
    感知器使用的激活函数就是一个sign符号函数
    :param x:
    :return:
    '''
    return 1 if x > 0 else -1


def pre(X, theta, b):
    y_ = X.dot(theta) + b
    return f(y_)


if __name__ == '__main__':
    # for i in range(10):
    #     X_train = [i for i in range(200)]
    #     rand_index = np.random.choice(200, size=20)
    #     print (rand_index)
    # batch_x = X_train[rand_index]
    # batch_ys = y_train[rand_index,:]

    X = [[3, 3], [4, 3], [1, 1], [3, 2], [3, 4], [2, 3]]
    y = [[1], [1], [-1], [1], [-1], [-1]]
    X = np.array(X, float)
    y = np.array(y, float)
    # plotData2(X,y)
    theta = np.zeros((X.shape[1], 1))
    b = 0
    theta, b = train(X, y, theta, b)
    plotData(X, y, b, theta)
    print(pre(np.array([[2, 2]]), theta, b))
