import operator
from math import log

# 训练样本创建
def createDataSet():
    dataSet = [['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
               ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
               ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
               ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
               ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
               ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
               ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
               ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],
               ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
               ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
               ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
               ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
               ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
               ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
               ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
               ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
               ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'], ]
    labels = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
    return dataSet, labels


def calE(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    E = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        E -= prob * log(prob, 2)
    return E


# 划分样本
#数据集，列数，特征
def divData(dataSet, axis, value):
    sub_data = []
    for e in dataSet:
        if e[axis] == value:
            feature = e[:axis]
            feature+=(e[axis + 1:])
            sub_data.append(feature)
    return sub_data


# 确定合适的样本划分标准
def chooseBestFeature(dataSet):
    n = len(dataSet[0]) - 1
    E = calE(dataSet)
    M_Gain = 0
    bestFeature = -1
    for i in range(n):
        L = [example[i] for example in dataSet]
        v_set = set(L)
        E2 = 0
        for e in v_set:
            sub_data = divData(dataSet, i, e)
            prob = len(sub_data) / float(len(dataSet))
            E2 += prob * calE(sub_data)
        Gain = E - E2
        if Gain > M_Gain:
            M_Gain = Gain
            bestFeature = i
    return bestFeature


# 当样本不可划分且判断结果不唯一时，同化确定结果
def selResult(res_data):
    count = {}
    for e in res_data:
        if e not in count.keys():
            count[e] = 0
        count[e] += 1
    A = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
    return A[0][0]


def createTree(dataSet, labels):
    res_data = [example[-1] for example in dataSet]
    if len(set(res_data)) == 1:
        return res_data[0]
    if len(dataSet[0]) == 1:
        return selResult(res_data)
    feature = chooseBestFeature(dataSet)
    fea_set = labels[feature]
    Tree = {fea_set: {}}
    del (labels[feature])
    fea_list = [example[feature] for example in dataSet]
    set_fea = set(fea_list)
    for value in set_fea:
        subLabels = labels[:]
        Tree[fea_set][value] = createTree(divData(dataSet, feature, value), subLabels)
    return Tree


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    Tree = createTree(dataSet, labels)
    print('决策树:')
    print(Tree)
    test_data = [['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘'],
                 ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑'],
                 ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑'],
                 ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘']]
    ideal_target = ['好瓜', '好瓜', '坏瓜', '坏瓜']
    labels = {'青绿': '色泽', '乌黑': '色泽', '浅白': '色泽', '蜷缩': '根蒂', '稍蜷': '根蒂', '硬挺': '根蒂', '清脆': '敲声', '浊响': '敲声',
              '沉闷': '敲声',
              '清晰': '纹理', '模糊': '纹理', '稍糊': '纹理', '凹陷': '脐部', '平坦': '脐部', '稍凹': '脐部', '软粘': '触感', '硬滑': '触感'}
    import Predict
    print('预测结果及准确率：')
    print(Predict.predict(Tree, test_data, labels, ideal_target))
