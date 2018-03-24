import numpy as np
import matplotlib.pyplot as plt

def strTOFloat(x):
    return float(x)

def computeCost(X,y,theta):
    '''

    :param X:X
    :param y:y
    :param theta:参数theta
    :return:cost值使用J表示
    '''
    m = len(y)
    J = 0
    H = np.dot(X,theta)
    J = (1.0 / (2 * m)) * (((H - y) ** 2).sum())
    return J

def gradientDescent(X, y, theta, alpha, iterations):
    '''

    :param X:X
    :param y:标签y
    :param theta:参数Theta
    :param alpha:学习率
    :param iterations:梯度下降的轮次
    :return:返回训练完成的参数Theta
    '''
    m = len(y)
    J_history = np.zeros(shape=[iterations,1])

    for i in range(iterations):
        H = np.dot(X,theta)
        temp = alpha * (1.0 / m) * ((H - y) * X).sum(axis=0)
        theta = theta - np.reshape(temp,[2,1])
        J_history[i] = computeCost(X,y,theta)

    return theta

def plotData(x,y,theta):

    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x,y,marker='x',color='red')
    print(theta[0][0],theta[1][0])

    xl = np.arange(-5.0,25,0.1)
    yl = theta[0][0] + theta[1][0] * xl
    plt.plot(xl,yl,color='black',label='linerRegression',linewidth='1')
    plt.show()


def pred(theta):
    predict1 = np.dot([[1, 3.5]],theta)
    print('For population = 35,000, we predict a profit of %f\n',predict1[0][0] * 10000);
    predict2 = np.dot([[1, 7]],theta)
    print('For population = 70,000, we predict a profit of %f\n',predict2[0][0] * 10000);

def main():

    alpha = 0.01
    iterations = 1500;

    train_list = [];
    with open('ex1data1.txt', 'r') as f:
        data = f.readlines()  # txt中所有字符串读入data
        for line in data:
            odom = line.split('\n')[0].split(',')  # 将单个数据分隔开存好
            numbers_float = list(map(strTOFloat, odom))  # 转化为浮点数
            train_list.append(numbers_float)

    train_set = np.array(train_list)
    m = train_set.shape[0] #数据集的数量
    n = train_set.shape[1] - 1 #数据集的维度

    X = train_set[:,0]#m*1
    y = np.reshape(train_set[:,1],[m,1]) #m*1

    b = np.array([np.ones(shape=[m,]),X]) #加入常数维度1，m*2
    b = b.transpose()
    theta = np.zeros(shape=[n+1,1])

    # b为输入函数中的X，y即为输出y
    cost = computeCost(b,y,theta)
    # 输出初始的cost值
    print(cost)
    theta = gradientDescent(b,y,theta,alpha=alpha,iterations=iterations)
    plotData(train_set[:,0],train_set[:,1],theta)
    pred(theta)

if __name__ == '__main__':
    main()



# print (np.shape(o))
# print (X)


