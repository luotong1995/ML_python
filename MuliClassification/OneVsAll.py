import numpy as np
import matplotlib.pyplot as plt
'''
对于多元分类，使用使用多个逻辑回归进行分类
对于训练好的额模型分别
'''


def strTOFloat(x):
    return float(x)

def sigmoid(x):
    return 1.0 / (1. + np.exp(-x))

def computeCost(X,y,theta,lamb):
    '''

    :param X:X
    :param y:y
    :param theta:参数theta
    :return:cost值使用J表示
    '''
    m = len(y)
    J = 0
    temp2 = (-y) * np.log(sigmoid(X.dot(theta)))
    temp3 = (1. - y) * np.log(1. - sigmoid(X.dot(theta)))
    temp = (temp2 - temp3)
    J = (1.0 / m) * temp.sum()
    rel = lamb / (2 * m) * ((theta ** 2).sum())
    rel = rel - lamb / (2 * m) * theta[0][0]
    J = J + rel
    return J

def computeAllCostAndGradientDescent(X,y,all_theta,lamb,alpha,iterations):

    m = len(y)
    for c in range(1,4):
        theta = np.zeros([X.shape[1],1])
        b = [1 if item[0] == c else 0 for item in y]
        b = np.array(b)
        b= b.reshape([len(b),1])
        theta = gradientDescent(X,b,theta,alpha,iterations,lamb)
        all_theta[c-1] = theta.reshape([1,theta.shape[0]])
    return all_theta

def gradientDescent(X, y, theta, alpha, iterations,lamb):
    '''

    :param X:X
    :param y:y
    :param theta:参数Theta
    :param alpha:学习率
    :param iterations:梯度下降的轮次
    :return:返回训练完成的参数Theta
    '''
    m = len(y)
    for i in range(iterations):
        temp = alpha * (1.0 / m) * ((sigmoid(X.dot(theta)) - y) * X).sum(axis=0)

        temp = np.reshape(temp, [temp.shape[0], 1])
        rel = lamb / m * theta
        # temp上增加正则
        temp = temp + rel
        temp[0][0] = (1 / m) * ((sigmoid(X.dot(theta)) - y) * X[:, 0]).sum()
        theta = theta - temp

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

    print(theta[0][0], theta[1][0])
    x = np.arange(np.min(x1), np.max(x2), 0.001)
    y = -1 / theta[2][0] * (theta[0][0] + theta[1][0] * x)
    plt.plot(x, y, color='black', label='linerRegression', linewidth='1')
    plt.show()


def pred():
    pass

def main():
    alpha = 0.001
    iterations = 50000;

    train_list = [];
    with open('iris.data.txt', 'r') as f:
        data = f.readlines()  # txt中所有字符串读入data
        for line in data:
            odom = line.split('\n')[0].split(',')  # 将单个数据分隔开存好
            # 处理label的的值，将字符串转换为可操作的类别数字
            if odom[-1] == 'Iris-setosa':
                odom[-1] = '1'
            elif odom[-1] == 'Iris-versicolor':
                odom[-1] = '2'
            elif odom[-1] == 'Iris-virginica':
                odom[-1] = '3'
            numbers_float = list(map(strTOFloat, odom))  # 转化为浮点数
            train_list.append(numbers_float)
    train_set = np.array(train_list,float)
    m = train_set.shape[0]  # 数据集的数量
    n = train_set.shape[1] - 1  # 数据集的维度

    X = train_set[:, 0:4]  # m*2
    # y = train_set[:,2]
    y = np.reshape(train_set[:, 4], [m, 1])  # m*1

    b = np.concatenate((np.ones([m,1]),X),axis=1) #加入常数维度  m*3
    all_theta = np.zeros(shape=[3,n + 1])

    # b为输入函数中的X，y即为输出y
    all_theta = computeAllCostAndGradientDescent(b, y, all_theta,lamb=1,alpha = alpha,iterations=iterations)
    print(all_theta)
    # plotData(theta)
    # pred(theta)


if __name__ == '__main__':
    main()
