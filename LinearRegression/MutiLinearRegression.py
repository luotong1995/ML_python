import numpy as np


# 使用matrix进行操作

def strTOFloat(x):
    return float(x)


def featureNormalize(X):
    '''
    做归一化
    :param X:
    :return:
    '''
    m = np.mean(X, 0)
    s = np.std(X, 0)
    X = X - m
    X = np.multiply(X, 1.0 / s)
    return X, m, s


def computeCostMulti(X, y, theta):
    '''

    :param X:X
    :param y:y
    :param theta:参数theta
    :return:cost
    '''
    m = len(y)
    J = 0
    H = X * theta
    J = (1.0 / (2 * m)) * ((np.power((H - y), 2)).sum())
    print(J)
    return J


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    '''

    :param X: x
    :param y: y
    :param theta: 参数theta
    :param alpha: 学习率
    :param num_iters: 训练轮次
    :return: 参数theta
    '''
    m = len(y)
    J_history = np.zeros([num_iters, 1])

    for i in range(num_iters):
        H = X * theta  # 矩阵相乘
        temp = alpha * (1.0 / m) * (np.multiply((H - y), X)).sum(axis=0)
        theta = theta - np.reshape(temp, [theta.shape[0], 1])
        J_history[i] = computeCostMulti(X, y, theta);

    return theta, J_history


# def pred(theta):

def main():
    alpha = 0.01
    iterations = 2100;

    train_list = [];
    with open('ex1data2.txt', 'r') as f:
        data = f.readlines()  # txt中所有字符串读入data
        for line in data:
            odom = line.split('\n')[0].split(',')  # 将单个数据分隔开存好
            numbers_float = list(map(strTOFloat, odom))  # 转化为浮点数
            train_list.append(numbers_float)

    train_set = np.array(train_list)
    m = train_set.shape[0]  # 数据集的数量
    n = train_set.shape[1] - 1  # X数据集的维度
    X = train_set[:, 0:2]
    # 归一化后的数据
    (X, mu, s) = featureNormalize(X)
    y = np.mat(train_set[:, 2]).reshape([m, 1])
    b = np.mat(np.concatenate((np.ones([m, 1]), X), axis=1))
    theta = np.mat(np.zeros([n + 1, 1]))
    computeCostMulti(b, y, theta)
    (theta, J) = gradientDescentMulti(b, y, theta, alpha, iterations)

    price = 0
    x_test = np.multiply(([1650, 3] - mu), 1.0 / s)
    x_test = np.mat(np.concatenate((np.ones([1, ]), x_test)))
    print('Predicted price of a 1650 sq-ft, 3 br house  is %f' % np.asarray(x_test * theta))


if __name__ == '__main__':
    main()
