from pandas import read_csv
from matplotlib import pyplot
import numpy as np

# read data from csv_file
series = read_csv('../../static/data_set.csv',
                  nrows=2000,
                  header=0,
                  parse_dates=[0],
                  index_col=0,
                  squeeze=True)

# initialize data to matplotlib format
series.plot()
# title of the graph
pyplot.title("Power Frequency Fluctuation from Transpower in  New Zealand in 2014")
# X,Y axis label
pyplot.xlabel("Time Sequence")
pyplot.ylabel("Frequency(HZ)")
# Set limit and ticks
pyplot.ylim(49.72, 50.23)
pyplot.yticks(np.arange(49.75, 50.25, step=0.05))
# Set fluctuation Baseline
pyplot.axhline(y=50.15, color='r')
pyplot.axhline(y=49.85, color='r')
# Print it to figure
# pyplot.savefig("../../export/initial_data_2000_row.png")
# display it
pyplot.show()
