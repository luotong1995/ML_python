from sklearn import svm
import matplotlib.pyplot as plt


def loadDataSet(file_name):
    data_mat = []
    label_mat = []
    with open(file_name, 'r') as f:
        # txt中所有字符串读入data
        data = f.readlines()
        for line in data:
            line_str = line.strip().split('\t')
            data_mat.append([float(line_str[0]), float(line_str[1])])
            label_mat.append(float(line_str[2]))
    return data_mat, label_mat


if __name__ == '__main__':
    x, y = loadDataSet('testSet.txt')
    clf = svm.SVC()
    clf.fit(x, y)
    clf.fit(x, y)
    print(clf.predict([[1.0, 1.0]]))
    # print(x)
    # print(y)
