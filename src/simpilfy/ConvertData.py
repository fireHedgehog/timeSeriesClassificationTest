from pandas import read_csv
from pandas import Series
from pandas import DataFrame

# read data from csv_file
series = read_csv('../../static/data_set.csv',
                  nrows=50,
                  header=0,
                  parse_dates=[0],
                  index_col=0,
                  squeeze=True)
raw_values = series.values

new_value = list()
for x in series:
    label = 0
    if x <= 49.85:
        label = -1
    elif x >= 50.15:
        label = 1
    else:
        label = 0
    new_value.append(label)

series_value = Series(new_value)
framed_data = DataFrame(series_value)

# framed_data.to_csv('../../export/labeled_data.csv', encoding='utf-8')

print(framed_data)
print(raw_values)
