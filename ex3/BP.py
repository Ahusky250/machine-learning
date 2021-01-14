import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from pandas.plotting import radviz


hide_num=22#隐藏层节点数量
feature=13#特征数量
label=3#标签数量
test_path='wine_test.csv'
train_path='wine_training.csv'
draw=False
learning_rate=0.0001
learning_num=100000


# 分类鸢尾花数据集时取消下列的注释
hide_num=10
feature=4
test_path='iris_test.csv'
train_path='iris_training.csv'
draw=True
learning_rate=0.4
learning_num=10000



# 1.初始化参数
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    # 权重和偏置矩阵
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))
    # 通过字典存储参数
    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    return parameters


# 2.前向传播
def forward_propagation(X, parameters):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    # 通过前向传播来计算a2
    z1 = np.dot(w1, X) + b1
    a1 = np.tanh(z1)  # 使用tanh作为第一层的激活函数
    z2 = np.dot(w2, a1) + b2
    a2 = 1 / (1 + np.exp(-z2))  # 使用sigmoid作为第二层的激活函数
    # 通过字典存储参数
    cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
    return a2, cache


# 3.计算代价函数
def compute_cost(a2, Y):
    m = Y.shape[1]  # Y的列数即为总的样本数

    # 采用交叉熵（cross-entropy）作为代价函数
    logprobs = np.multiply(np.log(a2), Y) + np.multiply((1 - Y), np.log(1 - a2))
    cost = - np.sum(logprobs) / m

    return cost


# 4.反向传播（计算代价函数的导数）
def backward_propagation(parameters, cache, X, Y):
    m = 120

    w2 = parameters['w2']

    a1 = cache['a1']
    a2 = cache['a2']

    # 反向传播，计算dw1、db1、dw2、db2
    dz2 = a2 - Y
    dw2 = (1 / m) * np.dot(dz2, a1.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.multiply(np.dot(w2.T, dz2), 1 - np.power(a1, 2))
    dw1 = (1 / m) * np.dot(dz1, X.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

    grads = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}

    return grads


# 5.更新参数
def update_parameters(parameters, grads, learning_rate):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    dw1 = grads['dw1']
    db1 = grads['db1']
    dw2 = grads['dw2']
    db2 = grads['db2']

    # 更新参数
    w1 = w1 - dw1 * learning_rate
    b1 = b1 - db1 * learning_rate
    w2 = w2 - dw2 * learning_rate
    b2 = b2 - db2 * learning_rate
    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    return parameters


# 建立神经网络
def nn_model(X, Y, n_h, n_input, n_output, num_iterations,learning_rate):
    np.random.seed(3)

    n_x = n_input  # 输入层节点数
    n_y = n_output  # 输出层节点数

    # 1.初始化参数
    parameters = initialize_parameters(n_x, n_h, n_y)

    # 梯度下降循环
    for i in range(0, num_iterations):
        # 2.前向传播
        a2, cache = forward_propagation(X, parameters)
        # 3.计算代价函数
        cost = compute_cost(a2, Y)
        # 4.反向传播
        grads = backward_propagation(parameters, cache, X, Y)
        # 5.更新参数
        parameters = update_parameters(parameters, grads,learning_rate)

        # 每1000次迭代，输出一次代价函数
        if i % 1000 == 0:
            print('迭代第%i次，代价函数为：%f' % (i, cost))

    return parameters


# 6.模型评估
def predict(parameters, x_test, y_test):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    z1 = np.dot(w1, x_test) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = 1 / (1 + np.exp(-z2))

    # 结果的维度
    n_rows = y_test.shape[0]
    n_cols = y_test.shape[1]

    # 预测值结果存储
    output = np.empty(shape=(n_rows, n_cols), dtype=int)

    for i in range(n_rows):
        for j in range(n_cols):
            if a2[i][j] > 0.5:
                output[i][j] = 1
            else:
                output[i][j] = 0

    print('预测结果：', output)
    print('真实结果：', y_test)

    count = 0
    for k in range(0, n_cols):
        if output[0][k] == y_test[0][k] and output[1][k] == y_test[1][k] and output[2][k] == y_test[2][k]:
            count = count + 1
        else:
            print('错误分类样本的序号：', k + 1)

    acc = count / int(y_test.shape[1]) * 100
    print('准确率：%.2f%%' % acc)
    return output


# 特征有4个维度，类别有1个维度，一共5个维度，故采用了RadViz图
def result_visualization(x_test, y_test, result):
    cols = y_test.shape[1]
    y = []
    pre = []

    # 反转换类别的独热编码
    for i in range(cols):
        if y_test[0][i] == 0 and y_test[1][i] == 0 and y_test[2][i] == 1:
            y.append('setosa')
        elif y_test[0][i] == 0 and y_test[1][i] == 1 and y_test[2][i] == 0:
            y.append('versicolor')
        elif y_test[0][i] == 1 and y_test[1][i] == 0 and y_test[2][i] == 0:
            y.append('virginica')

    for j in range(cols):
        if result[0][j] == 0 and result[1][j] == 0 and result[2][j] == 1:
            pre.append('setosa')
        elif result[0][j] == 0 and result[1][j] == 1 and result[2][j] == 0:
            pre.append('versicolor')
        elif result[0][j] == 1 and result[1][j] == 0 and result[2][j] == 0:
            pre.append('virginica')
        else:
            pre.append('未知种类')

    # 将特征和类别矩阵拼接起来
    real = np.column_stack((x_test.T, y))
    prediction = np.column_stack((x_test.T, pre))

    df_real = pd.DataFrame(real, index=None,
                           columns=['萼片长度', '萼片宽度', '花瓣长度', '花瓣宽度', '种类'])
    df_prediction = pd.DataFrame(prediction, index=None,
                                 columns=['萼片长度', '萼片宽度', '花瓣长度', '花瓣宽度', '种类'])

    df_real[['萼片长度', '萼片宽度', '花瓣长度', '花瓣宽度']] = df_real[
        ['萼片长度', '萼片宽度', '花瓣长度', '花瓣宽度']].astype(float)
    df_prediction[['萼片长度', '萼片宽度', '花瓣长度', '花瓣宽度']] = df_prediction[
        ['萼片长度', '萼片宽度', '花瓣长度', '花瓣宽度']].astype(float)

    # 绘图

    plt.figure('真实分类')
    radviz(df_real, '种类', color=['blue', 'green', 'red', 'yellow'])
    plt.figure('预测分类')
    radviz(df_prediction, '种类', color=['blue', 'green', 'red', 'yellow'])
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()


if __name__ == "__main__":
    # 读取数据
    data_set = pd.read_csv(train_path, header=None)
    X = data_set.iloc[:, 0:feature].values.T
    Y = data_set.iloc[:, feature:].values.T
    Y = Y.astype('uint8')

    # 开始训练
    parameters = nn_model(X, Y, n_h=hide_num, n_input=feature, n_output=label, num_iterations=learning_num,learning_rate=learning_rate)

    # 对模型进行测试
    data_test = pd.read_csv(test_path, header=None)
    x_test = data_test.iloc[:, 0:feature].values.T
    y_test = data_test.iloc[:, feature:].values.T
    y_test = y_test.astype('uint8')

    result = predict(parameters, x_test, y_test)

    # 分类结果可视化(红酒数据集特征数量过多，不绘制)
    if draw:
        result_visualization(x_test, y_test, result)
