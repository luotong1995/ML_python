import math
import operator


def calcShannonEnt(x, y):
    '''
    计算香农熵
    :param x:
    :param y:
    :return:
    '''

    length = len(x)
    lable_dic = {}
    for item in y:
        if item not in lable_dic:
            lable_dic[item] = 1
        else:
            lable_dic[item] += 1
    ent = 0.0
    for item in lable_dic.keys():
        prob = float(lable_dic[item]) / length
        ent -= prob * math.log(prob, 2)
    return ent


def splitDataSet(x, y, axis, value):
    '''
    划分出axis维特征为value的所有的数据，其实为一个条件概率情况
    :param x:
    :param y:
    :param axis:作为划分数据集的属性下标
    :param value:划分数据集的属性值
    :return:划分之后除去本axis这一列的新的数据集
    '''
    new_x = []
    new_y = []
    for index, item in enumerate(x):
        if item[axis] == value:
            # 去除axis这一列
            x_ = item[:axis]
            x_.extend(item[axis + 1:])
            new_x.append(x_)
            new_y.append(y[index])
    return new_x, new_y


def chooseBestFeatureToSplit(x, y):
    '''
    根据信息增益找到最好的属性
    :param x:
    :param y:
    :return:
    '''
    all_ent = calcShannonEnt(x, y)
    num_features = len(x[0])
    best_gain = 0.0
    best_axis = 0
    for i in range(num_features):
        features = [item[i] for item in x]
        feature_set = set(features)
        now_ent = 0.0
        for value in feature_set:
            sub_x, sub_y = splitDataSet(x, y, i, value)
            prob = len(sub_x) / len(x)
            now_ent += prob * calcShannonEnt(sub_x, sub_y)
        infoGain = all_ent - now_ent
        if infoGain > best_gain:
            best_gain = infoGain
            best_axis = i
    return best_axis


def class_vote(y):
    '''

    :param y:分类标签
    :return:返回最多的那一类
    '''
    class_count = {}
    for item in y:
        if item not in class_count:
            class_count[item] = 1
        else:
            class_count[item] += 1
    sorted_cls = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_cls[0][0]


def creatTree(x, y, attribute_names):
    '''

    :param x:输入
    :param y:分类lable
    :param attribute_names:属性名 默认为None
    :return:构建好的树
    '''

    # 如果此时的y中都是一个类，就直接返回该类
    if all(y[0] == item for item in y):
        return y[0]
    # 如果遍历完所有的属性，则返回此时y中类别最多的那个
    if len(x) == 0:
        return class_vote(y)
    best_feture = chooseBestFeatureToSplit(x, y)
    best_feture_names = attribute_names[best_feture]
    new_attr_names = attribute_names[:]
    del (new_attr_names[best_feture])
    myTree = {best_feture_names: {}}
    features_value = [item[best_feture] for item in x]
    # 根据不同的属性value构建树的分支
    for value in set(features_value):
        sub_attribute_names = new_attr_names[:]
        sub_x, sub_y = splitDataSet(x, y, best_feture, value)
        myTree[best_feture_names][value] = creatTree(sub_x, sub_y, sub_attribute_names)
    return myTree


def createDataSet():
    '创建测试数据集'
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]

    x = []
    y = []
    for item in dataSet:
        x.append(item[0:-1])
        y.append(item[-1])
    return x, y


def classify(inputTree, attribute_names, x_test):
    firstStr = list(inputTree.keys())[0]
    feature_index = attribute_names.index(firstStr)
    sec_dict = inputTree[firstStr]

    for key in sec_dict.keys():
        if x_test[feature_index] == key:
            if type(sec_dict[key]) == dict:
                class_label = classify(sec_dict[key], attribute_names, x_test)
            else:
                class_label = sec_dict[key]
    return class_label


if __name__ == '__main__':
    x, y = createDataSet()
    # attribute_names
    attribute_names = []
    for i in range(len(x[0])):
        attribute_names.append(str(i))
    mytree = creatTree(x, y, attribute_names)
    print(classify(mytree, attribute_names, [1, 1]))
