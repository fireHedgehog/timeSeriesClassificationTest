from pandas import read_csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.tree import export_graphviz
from subprocess import call


# read data from csv_file
def read_data():
    series = read_csv('../../static/data_set.csv', nrows=30000, parse_dates=["date"])
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
X_features = np.delete(X_features, (len(X_features) - 1), axis=0)  # make data size consistent

print(y_true)
print(X_features)

clf = RandomForestClassifier(random_state=14)
scores = cross_val_score(clf, X_features, y_true, scoring='accuracy')

clf.fit(X_features, y_true)

print("Cross validation")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))  # Accuracy: 73.2%

y_importances = clf.feature_importances_
x_importances = tested_feature
y_pos = np.arange(len(x_importances))

plt.barh(y_pos, y_importances, align='center')
plt.yticks(y_pos, x_importances)
plt.xlabel('Importance')
plt.xlim(0, 1)
plt.title("Features' Importance")
plt.show()

plt.bar(y_pos, y_importances, width=0.4, align='center', alpha=0.4)
plt.xticks(y_pos, x_importances)
plt.ylabel('Importance')
plt.ylim(0, 1)
plt.title("Features' Importance")
plt.show()

# tree graph
# estimator = clf.estimators_[5]

# Export as dot file
# export_graphviz(estimator,out_file='tree.dot', feature_names=tested_feature, class_names=y_true, rounded=True,  proportion=False, precision=2, filled=True)

# call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
