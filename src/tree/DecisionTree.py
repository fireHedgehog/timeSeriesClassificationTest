from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from pandas import read_csv
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn import tree
# import pydotplus

data = read_csv('../../static/featured_data_v4.csv', parse_dates=["date"], skiprows=range(1, 2))

tested_feature = ["difference", "previous", "second", "threshold_alarm", "week_day"]
x = data[tested_feature].values
y = data['fluctuation_type'].values

# y = np.delete(y, 0, axis=0)  # move 1 T ahead
# x = np.delete(x, (len(x) - 1), axis=0)  # make data size consistent

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.9)

clf = DecisionTreeClassifier(random_state=14)
# clf = svm.SVC(C=0.8, kernel='rbf', decision_function_shape='ovr')
clf.fit(x_train, y_train.ravel())

scores_train = clf.score(x_train, y_train)
y_pre_train = clf.predict(x_train)

scores_test = clf.score(x_test, y_test)
y_pre_test = clf.predict(x_test)

print("\nTrain Accuracy: {0:.1f}%".format(np.mean(scores_train) * 100))
# print("\nTrain Predication: \n", y_pre_train)

print("\nTest Accuracy: {0:.1f}%".format(np.mean(scores_test) * 100))
# print("\ntest Predication: \n", y_pre_test)

# print('decision_function:\n', clf.decision_function(x_train))
matrix = confusion_matrix(y_test, y_pre_test, labels=['No', 'Lower', 'Upper'])
print(matrix)
print(classification_report(y_test, y_pre_test))

sns.heatmap(matrix, annot=True)
plt.show()

# dot_data = tree.export_graphviz(clf,
#                                 out_file=None,
#                                 feature_names=tested_feature,
#                                 class_names=y_train.ravel(),
#                                 filled=True,
#                                 rounded=True,
#                                 special_characters=True
#                                 )
#
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_pdf("iris.pdf")

# 画图
# x_min, x_max = X_features[:, 0].min() - 1, X_features[:, 0].max() + 1
# y_min, y_max = y_true[:, 1].min() - 1, y_true[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
#                      np.arange(y_min, y_max, 0.1))
#
# Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
#
# plt.contourf(xx, yy, Z, alpha=0.4)
# plt.scatter(X_features[:, 0], X_features[:, 1], c=y_true, alpha=0.8)
# plt.show()
