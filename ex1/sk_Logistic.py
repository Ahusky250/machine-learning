from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# 从sklearn中获取iris数据集，写入本地文件中
dataset = load_iris()
# 构建训练集
# 构建测试集及测试集对应的正确答案
train_data, test_data, train_target, test_target = train_test_split(dataset.data, dataset.target, test_size=0.3,
                                                                    random_state=0, stratify=dataset.target)
# 开始训练
clf = LogisticRegression(max_iter=100)
clf = clf.fit(train_data, train_target)

# 开始测试
print('Coefficients :%s,intercept %s' % (clf.coef_, clf.intercept_))
print('Score: %.2f' % clf.score(test_data, test_target))
print(classification_report(test_target, clf.predict(test_data)))