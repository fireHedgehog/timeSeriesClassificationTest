from pandas import read_csv
from pandas import Series
from pandas import DataFrame

# read data from csv_file
series = read_csv('../../static/data_set.csv', parse_dates=["date"])

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

series_value = Series(series)
framed_data = DataFrame(series_value)

#framed_data.to_csv('../../export/featured_data.csv', encoding='utf-8')
