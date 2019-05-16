from pandas import read_csv
from pandas import DataFrame

# read data from csv_file
series = read_csv('../../static/data_set.csv', parse_dates=["date"])
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

framed_data = DataFrame(series)

framed_data.to_csv('../../export/featured_data.csv', encoding='utf-8')
