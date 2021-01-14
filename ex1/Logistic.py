"""
对数几率回归算法的实现及使用
"""

import numpy as np
from sklearn.datasets import load_iris


def logicFun(z):
    return 1.0 / (1 + np.exp(-z))


def trainning(dataset, labelset, test_data, test_label):
    # 将列表转化为矩阵
    data = np.mat(dataset)
    label = np.mat(labelset).transpose()

    # 初始化参数列表coef
    coef = np.ones((len(dataset[0]) + 1, 1))

    # 属性矩阵最后添加一列全1列（参数w中有常数参数）
    a = np.ones((len(dataset), 1))
    data = np.c_[data, a]

    # 步长
    n = 0.1

    # 每次迭代计算一次正确率（在测试集上的正确率）
    rightrate = 0.0
    cnt=0
    while rightrate < 0.8:
        cnt=cnt+1
        # 计算当前参数coef下的预测值
        c = logicFun(np.dot(data, coef))

        # 梯度下降的计算过程，对照着梯度下降的公式
        b = c - label
        change = np.dot(np.transpose(data), b)
        coef = coef - change * n

        # 预测，更新正确率
        rightrate, pre = test(test_data, test_label, coef)
    return coef,cnt


def test(dataset, labelset, coef):
    data = np.mat(dataset)
    a = np.ones((len(dataset), 1))
    data = np.c_[data, a]

    # 使用训练好的参数coef进行计算
    y = logicFun(np.dot(data, coef))
    b, c = np.shape(y)

    # 记录预测正确的个数，用于计算正确率
    rightcount = 0
    pre = []
    for i in range(b):

        # 预测标签
        flag = -1

        # 大于0.5的为正例
        if y[i, 0] > 0.5:
            flag = 1

        # 小于等于0.5的为反例
        else:
            flag = 0

        pre.append(flag)

        # 记录预测正确的个数
        if labelset[i] == flag:
            rightcount += 1

    # 正确率
    rightrate = rightcount / len(dataset)
    return rightrate, pre


if __name__ == '__main__':
    # 构建测试集及测试集对应的正确答案
    # 色泽：青绿-1，乌黑-2，浅白-3
    # 根蒂：蜷缩-1，稍卷-2，硬挺-3
    # 敲声：浊响-1，沉闷-2，清脆-3
    # 纹理：清晰-1，稍糊-2，模糊-3
    # 脐部：凹陷-1，稍凹-2，平坦-3
    # 触感：硬滑-1，软粘-2
    # [色泽，根蒂，敲声，纹理，脐部，触感]
    train_data = [
        [1, 1, 1, 1, 1, 1],
        [2, 1, 2, 1, 1, 1],
        [2, 1, 1, 1, 1, 1],
        [1, 2, 1, 1, 2, 2],
        [2, 2, 1, 2, 2, 2],
        [1, 3, 3, 1, 3, 2],
        [3, 2, 2, 2, 1, 1],
        [2, 2, 1, 1, 2, 2],
        [3, 1, 1, 3, 3, 1],
        [1, 1, 2, 2, 2, 1]
    ]
    train_label = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    test_data = [
        [1, 1, 2, 1, 1, 1],
        [3, 1, 1, 1, 1, 1],
        [2, 2, 1, 1, 2, 1],
        [2, 2, 2, 2, 2, 1],
        [3, 3, 3, 3, 3, 1],
        [3, 1, 1, 3, 3, 2],
        [1, 2, 1, 1, 1, 1]
    ]
    ideal_label = [1, 1, 1, 0, 0, 0, 0]
    coef,cnt = trainning(train_data, train_label, test_data, ideal_label)
    print('用于预测的函数的各项系数为：\n', coef)
    print('迭代了：',cnt,'次')
    rightrate, pre = test(test_data, ideal_label, coef)
    print('预测精度为：\t', rightrate)
    print('预测结果为：\t', pre)
    print('正确结果为：\t', ideal_label)
