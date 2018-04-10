import numpy as np
import matplotlib.pyplot as plt

def strTOFloat(x):
    return float(x)

def sigmoid(x):
    return 1.0 / (1. + np.exp(-x))

def computeCost(X,y,theta):
    '''

    :param X:X
    :param y:y
    :param theta:参数theta
    :return:cost值使用J表示
    '''
    m = len(y)
    J = 0
    temp1 = sigmoid(X.dot(theta))

    temp2 = (-y) * np.log(temp1)
    temp3 = (1. - y) * np.log(1. - temp1)
    temp = (temp2 - temp3)
    J = (1.0 / m) * temp.sum()
    return J

def gradientDescent(X, y, theta, alpha, iterations):
    '''

    :param X:X
    :param y:y
    :param theta:参数Theta
    :param alpha:学习率
    :param iterations:梯度下降的轮次
    :return:返回训练完成的参数Theta
    '''
    m = len(y)
    J_history = np.zeros(shape=[iterations, 1])

    for i in range(iterations):
        temp = alpha * (1.0 / m) * ((sigmoid(X.dot(theta)) - y) * X).sum(axis=0)
        theta = theta - np.reshape(temp, [3, 1])
        J_history[i] = computeCost(X,y,theta)
    return theta

def plotData(x1, x2,y, theta):
    plt.xlabel('x1')
    plt.ylabel('x2')
    m = len(y)
    for i in range(m):
        if int(y[i]) == 1:
            plt.scatter(x1[i], x2[i], marker='x', color='red')
        else:
            plt.scatter(x1[i], x2[i], marker='o', color='blue')

    print('theta', theta[0][0], theta[1][0],theta[2][0])
    xl = np.arange(np.min(x1), np.max(x2), 0.001)
    yl = -1 / theta[2][0] * (theta[0][0] + theta[1][0] * xl)
    plt.plot(xl, yl, color='black', label='linerRegression', linewidth='1')
    plt.show()


def pred():
    pass

def main():
    alpha = 0.001
    iterations = 1000000;

    train_list = [];
    with open('ex2data1.txt', 'r') as f:
        data = f.readlines()  # txt中所有字符串读入data
        for line in data:
            odom = line.split('\n')[0].split(',')  # 将单个数据分隔开存好
            numbers_float = list(map(strTOFloat, odom))  # 转化为浮点数
            train_list.append(numbers_float)

    train_set = np.array(train_list,float)
    m = train_set.shape[0]  # 数据集的数量
    n = train_set.shape[1] - 1  # 数据集的维度

    X = train_set[:, 0:2]  # m*2
    # y = train_set[:,2]
    y = np.reshape(train_set[:, 2], [m, 1])  # m*1

    #
    b = np.concatenate((np.ones([m,1]),X),axis=1) #加入常数维度  m*3
    theta = np.zeros(shape=[n + 1, 1])

    # b为输入函数中的X，y即为输出y
    cost = computeCost(b, y, theta)
    # 输出初始的cost值
    print('init cost' + str(cost))
    theta = gradientDescent(b, y, theta, alpha=alpha, iterations=iterations)
    # print(theta)
    # plotData(theta)
    plotData(train_set[:, 0], train_set[:, 1],train_set[:,2], theta)
    # pred(theta)


if __name__ == '__main__':
    main()
    # print(sigmoid(np.inf))