from pandas import read_csv
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from pandas import DataFrame


# read data from csv_file
def read_data():
    series = read_csv('../../static/data_set.csv', nrows=60000, parse_dates=["date"])
    # add new feature
    series["week_day"] = ""
    series["hour"] = "1"
    series["fluctuation"] = False
    series["fluctuation_type"] = ""  # fluctuation type 0 means no, 1 upper, -1 means lower
    series["trend"] = 1  # 0 means stable, 1 means increasing, -1 means decreasing
    series["difference"] = 0  # changing value
    previous_value = 0  # check the trend

    for index, row in series.iterrows():

        # add new feature
        row["week_day"] = row["date"].weekday()
        row['hour'] = row["date"].hour  # time

        # check trend
        if previous_value <= row["frequency"]:
            row["trend"] = 1
        else:
            row["trend"] = -1

        row["difference"] = row["frequency"] - previous_value
        previous_value = row["frequency"]

        # check lower or upper
        label = 'No'
        has = False
        if row["frequency"] <= 49.85:
            label = 'Lower'
            has = True
        elif row["frequency"] >= 50.15:
            label = 'Upper'
            has = True
        row['fluctuation'] = has  # fluctuation
        row['fluctuation_type'] = label  # fluctuation

        series.iloc[index] = row
    return series


series_data = read_data()
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

# 画图
x_min, x_max = X_features[:, 0].min() - 1, X_features[:, 0].max() + 1
y_min, y_max = y_true[:, 1].min() - 1, y_true[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_features[:, 0], X_features[:, 1], c=y_true, alpha=0.8)
plt.show()
