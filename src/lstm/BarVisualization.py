import numpy as np
from matplotlib import pyplot
from pandas import DataFrame
from pandas import read_csv
from pandas import concat

# read data from csv_file
series = read_csv('../../static/data_set_labeled.csv',
                 nrows=2000,
                  header=0,
                  parse_dates=[0],
                  index_col=0,
                  squeeze=True)

# Create lagged dataset
values = DataFrame(series.values)
series.plot()

# title of the graph
pyplot.title("Power Frequency Fluctuation from Transpower in  New Zealand in 2014")
# X,Y axis label
pyplot.xlabel("Time Sequence")
pyplot.ylabel("Fluctuation")
# Set limit and ticks
pyplot.ylim(-1.2, 1.2)
pyplot.yticks(np.arange(-1.5, 1.5, step=1), labels=['', '<=49.85', '>=50.15'])
# Set fluctuation Baseline
pyplot.axhline(y=0, color='black', lw=2)
# Print it to figure
# pyplot.savefig("../../export/row_data_labeled.png")
# display it
# rects1 = plt.bar(left=x, height=num_list1, width=0.4, alpha=0.8, color='red', label="一部门")
pyplot.show()
