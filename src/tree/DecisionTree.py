from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# read_featured_data
series_data = read_csv('../../static/featured_data.csv', parse_dates=["date"])

tested_feature = ["week_day", "hour", "difference", "frequency"]

y_true = series_data['fluctuation_type'].values
X_features = series_data[tested_feature].values

y_true = np.delete(y_true, 0, axis=0)  # move 1 T ahead
y_true = np.delete(y_true, 0, axis=0)  # move 1 T ahead

X_features = np.delete(X_features, (len(X_features) - 1), axis=0)  # make data size consistent
X_features = np.delete(X_features, (len(X_features) - 1), axis=0)  # make data size consistent

print(y_true)
print(X_features)

clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X_features, y_true, scoring='accuracy')

clf.fit(X_features, y_true)

print("Cross validation")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))  # Accuracy: 73.2%git

print(confusion_matrix(y_true, y_true))

C2 = confusion_matrix(y_true, y_true)
sns.heatmap(C2, annot=True)
plt.show()

# dot_data = tree.export_graphviz(clf, out_file=None,
#                                 feature_names=tested_feature,
#                                 class_names=y_true,
#                                 filled=True, rounded=True,
#                                 special_characters=True)
#
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_pdf("iris.pdf")

# # 画图
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
