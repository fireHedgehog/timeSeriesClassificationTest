from pandas import read_csv
from pandas import DataFrame


def t2s(t):
    return int(t.hour) * 3600 + int(t.minute) * 60 + int(t.second)


# read data from csv_file
series = read_csv('../../static/featured_data_v2.csv', parse_dates=["date"])
# add new feature
series["second"] = 0
series["alarm"] = False  # 0 means stable, 1 means increasing, -1 means decreasing
series["difference"] = 0  # changing value

previous_value = 0  # check the trend

for index, row in series.iterrows():

    row['second'] = t2s(row["date"])  # second

    # check trend
    if row["frequency"] >= 50.10 or row["frequency"] <= 49.90:
        row["alarm"] = True
    else:
        row["alarm"] = False

    row["difference"] = round((row["previous"] - previous_value), 2)
    previous_value = row["previous"]
    series.iloc[index] = row

framed_data = DataFrame(series)

framed_data.to_csv('../../export/featured_data_v3.csv', encoding='utf-8')
