from pandas import read_csv
from pandas import DataFrame
from pandas import Series


# read data from csv_file
series = read_csv('../../static/featured_data_v3.csv', parse_dates=["date"])

previous_value = 0  # check the trend
new_value = list()
for index, row in series.iterrows():
    alarm = 0
    if row["previous"] >= 50.10:
        alarm = 1
    elif row["previous"] <= 49.90:
        alarm = 2
    else:
        alarm = 0

    previous_value = row["previous"]
    new_value.append(alarm)

series_value = Series(new_value)
framed_data = DataFrame(series_value)

# framed_data.to_csv('../../export/labeled_data_2.csv', encoding='utf-8')
