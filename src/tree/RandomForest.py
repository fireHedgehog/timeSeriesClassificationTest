from pandas import read_csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# read data from csv_file
series = read_csv('../../static/data_set.csv', nrows=10000, parse_dates=["date"])

# add new feature
series["week_day"] = ""
series["day_time"] = "1"
series["fluctuation"] = False
series["fluctuation_type"] = ""  # fluctuation type 0 means no, 1 upper, -1 means lower
series["trend"] = 1  # 0 means stable, 1 means increasing, -1 means decreasing
series["abs_change"] = 0  # changing value
previous_value = 0  # check the trend

for index, row in series.iterrows():

    # add new feature
    row["week_day"] = row["date"].weekday()
    # check trend
    if previous_value <= row["frequency"]:
        row["trend"] = 1
    else:
        row["trend"] = -1

    row["abs_change"] = abs((previous_value - row["frequency"]))
    previous_value = row["frequency"]

    # check lower or upper
    label = 0
    has = False
    if row["frequency"] <= 49.85:
        label = 1
        has = True
    elif row["frequency"] >= 50.15:
        label = 1
        has = True
    row['fluctuation'] = has  # fluctuation
    row['fluctuation_type'] = label  # fluctuation

    # check day_time
    time = 1
    if 0 <= row["date"].hour <= 8:
        time = 1
    elif 8 < row["date"].hour <= 16:
        time = 2
    else:
        time = 3
    row['day_time'] = time  # time

    series.iloc[index] = row

y_true = series['fluctuation'].values
X_features = series[["week_day", "day_time", "abs_change"]].values

clf = RandomForestClassifier(random_state=14)
scores = cross_val_score(clf, X_features, y_true, scoring='accuracy')

clf.fit(X_features, y_true)

print("Using just the last result from the home and visitor teams")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

y_importances = clf.feature_importances_
x_importances = ["week_day", "day_time", "abs_change"]
y_pos = np.arange(len(x_importances))
# 横向柱状图
plt.barh(y_pos, y_importances, align='center')
plt.yticks(y_pos, x_importances)
plt.xlabel('Importances')
plt.xlim(0, 1)
plt.title('Features Importances')
plt.show()

# 竖向柱状图
plt.bar(y_pos, y_importances, width=0.4, align='center', alpha=0.4)
plt.xticks(y_pos, x_importances)
plt.ylabel('Importances')
plt.ylim(0, 1)
plt.title('Features Importances')
plt.show()
# sns.pointplot(x='day_time', y='fluctuation', hue='week_day', data=series)

# plt.show()
# print(series)
