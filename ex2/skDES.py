import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import matplotlib.pyplot as plt

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
feature_name = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
feature = []
labels = []
for e in dataSet:
    feature.append(e[0:-1])
    labels.append(e[-1])

to1 = ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑']
to2 = ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '软粘']
to3 = ['浅白', '硬挺', '清脆', '模糊', '平坦']
for e in feature:
    for i in range(len(e)):
        if e[i] in to1:
            e[i] = 1
        elif e[i] in to2:
            e[i] = 2
        else:
            e[i] = 3
for i in range(len(labels)):
    if labels[i] == '好瓜':
        labels[i] = 1
    else:
        labels[i] = 0
test_feature=[['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘'],
                 ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑'],
                 ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑'],
                 ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘']]
for e in test_feature:
    for i in range(len(e)):
        if e[i] in to1:
            e[i] = 1
        elif e[i] in to2:
            e[i] = 2
        else:
            e[i] = 3
test_labels=[1,1,0,0]
clf = DecisionTreeRegressor(max_depth=len(feature[0]))
clf.fit(feature, labels)
print('预测结果：')
print([int(i) for i in clf.predict(test_feature)])
print('正确结果：')
print(test_labels)

export_graphviz(
    clf,
    out_file="D:\\tree.dot",
    feature_names=feature_name,
    rounded=True,
    filled=True
    #dot -Tpng tree.dot -o tree.png
)
