from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
from pandas import read_csv
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = read_csv('../../static/featured_data_v3.csv', parse_dates=["date"], skiprows=range(1, 2))

tested_feature = ["trend", "difference", "previous", "second", "alarm", "week_day"]
x = data[tested_feature].values
y = data['fluctuation'].values

# y = np.delete(y, 0, axis=0)  # move 1 T ahead
#
# x = np.delete(x, (len(x) - 1), axis=0)  # make data size consistent

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.88)

clf = svm.SVC(C=0.5, kernel='linear', decision_function_shape='ovr')
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
matrix = confusion_matrix(y_test, y_pre_test, labels=[True, False])
print(matrix)
print(classification_report(y_test, y_pre_test))

sns.heatmap(matrix, annot=True)
plt.show()

for i in x_train:
    res = clf.predict(np.array(i).reshape(1, -1))
if res[0]:
    plt.scatter(i[0], i[1], c='r', marker='*')
else:
    plt.scatter(i[0], i[1], c='g', marker='*')

for i in x_test:
    res = clf.predict(np.array(i).reshape(1, -1))
if res[0]:
    plt.scatter(i[0], i[1], c='r', marker='.')
else:
    plt.scatter(i[0], i[1], c='g', marker='.')
plt.show()
