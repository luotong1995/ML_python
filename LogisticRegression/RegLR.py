import numpy as np
import matplotlib.pyplot as plt

degree = 6

def strTOFloat(x):
    '''
    字符串转换为Float
    :param x:
    :return:
    '''
    return float(x)

def sigmoid(x):
    '''
    sigmoid函数
    :param x:
    :return:
    '''
    return 1.0 / (1. + np.exp(-x))

def feature_map(x1,x2,degree):
    '''
    该数据集不是线性可分的，所以增加参数，增加特征的维度
    :param X:
    :param degree:
    :return:
    '''
    if isinstance(x1,float):
        X = [1]
        for i in range(1,degree+1):
            for j in range(i+1):
                # temp1 = (np.array([X[:,0]]) ** (i - j))
                # temp2 = (np.array([X[:,1]]) ** j)
                temp1 = (x1 ** (i - j))
                temp2 = (x2 ** j)
                temp = temp1 * temp2
                X.append(temp)
        X = np.array(X).reshape([1,len(X)])
        return X
    else:
        X = np.ones([x1.shape[0],1])
        for i in range(1,degree+1):
            for j in range(i+1):
                # temp1 = (np.array([X[:,0]]) ** (i - j))
                # temp2 = (np.array([X[:,1]]) ** j)
                temp1 = (np.array([x1]) ** (i - j))
                temp2 = (np.array([x2]) ** j)
                temp = temp1 * temp2
                X = np.concatenate((X,temp.transpose()), axis=1)
        return X

def computeCost(X, y, theta,lamb):
    '''
    计算Cost
    :param X: X
    :param y: y
    :param theta:参数
    :param lamb:正则参数
    :return: cost
    '''
    m = len(y)
    J = 0
    H = sigmoid(X.dot(theta))

    # temp2 = (-y) * np.log(H)
    # temp3 = (1. - y) * np.log(1. - H)
    temp = ((-y) * np.log(H) - (1. - y) * np.log(1. - H))
    # 计算从theta0 - theta N 的正则
    rel = lamb / (2*m) * ((theta ** 2).sum())
    # 本该不计算在其中，减去theta0计算的正则
    rel  = rel - lamb / (2*m) * theta[0][0]
    J = (1.0 / m) * temp.sum() + rel
    return J

def gradientDescent(X, y, theta, alpha, iterations,lamb):
    '''

    :param X:X
    :param y:y
    :param theta:参数Theta
    :param alpha:学习率
    :param iterations:梯度下降的轮次
    :param lamb:正则参数
    :return:返回训练完成的参数Theta
    '''
    m = len(y)
    # J_history = np.zeros(shape=[iterations, 1])

    for i in range(iterations):
        temp = (1.0 / m) * ((sigmoid(X.dot(theta)) - y) * X).sum(axis=0)

        temp = np.reshape(temp,[temp.shape[0],1])
        rel = lamb / m * theta
        # temp上增加正则
        temp = temp + rel
        temp[0][0] = (1.0 / m) * ((sigmoid(X.dot(theta))-y) * X[:,0]).sum()
        theta = theta - alpha * temp

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

    # x_map = np.arange(np.min(x1), np.max(x2), 0.001)
    # y_map = np.arange(np.min(x1), np.max(x2), 0.001)

    x_map = np.linspace(-1,1.5,50)
    y_map = np.linspace(-1, 1.5, 50)
    X, Y = np.meshgrid(x_map, y_map)
    z = np.zeros([x_map.shape[0],x_map.shape[0]])
    for i in range(x_map.shape[0]):
        for j in range(y_map.shape[0]):
            z[i][j] = feature_map(x_map[i],y_map[j],degree).dot(theta)


    plt.contour(X,Y,z,colors = 'black', linewidth = 0.5)
    # yl = -1 / theta[2][0] * (theta[0][0] + theta[1][0] * xl)
    # plt.plot(xl, yl, color='black', label='linerRegression', linewidth='1')
    plt.show()

def predAndAcc(X,y,theta):
    temp = sigmoid(X.dot(theta))
    p = np.array([1 if i > 0.5 else 0 for i in temp])
    y = np.array([int(i[0]) for i in y])
    print(p)
    acc = np.mean(p==y)
    return acc

def main():
    alpha = 0.001
    iterations = 200000

    train_list = []
    with open('ex2data2.txt', 'r') as f:
        data = f.readlines()  # txt中所有字符串读入data
        for line in data:
            odom = line.split('\n')[0].split(',')  # 将单个数据分隔开存好
            numbers_float = list(map(strTOFloat, odom))  # 转化为浮点数
            train_list.append(numbers_float)

    train_set = np.array(train_list,float)
    m = train_set.shape[0]  # 数据集的数量

    X = train_set[:, 0:2]  # m*2
    # y = train_set[:,2]
    y = np.reshape(train_set[:, 2], [m, 1])  # m*1
    # feature map 增多特征维度
    X = feature_map(X[:,0],X[:,1],degree=degree)
    print (X.shape[1])
    n = X.shape[1]  # 数据集的维度
    # 定义Theta初始值
    theta = np.zeros(shape=[n, 1])

    # X为输入，y即为输出y,theta为参数
    cost = computeCost(X, y, theta,lamb=1)
    print (cost)
    theta = gradientDescent(X, y, theta, alpha=alpha, iterations=iterations,lamb=1)
    print (theta)
    plotData(train_set[:, 0], train_set[:, 1],train_set[:,2], theta)
    print (predAndAcc(X,y,theta))


if __name__ == '__main__':
    main()