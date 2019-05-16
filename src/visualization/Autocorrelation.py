from pandas import read_csv
from statsmodels.graphics.tsaplots import plot_acf

from matplotlib import pyplot

series = read_csv('../../static/data_set.csv',
                  nrows=2000,
                  header=0,
                  parse_dates=[0],
                  index_col=0,
                  squeeze=True)
plot_acf(series)

pyplot.show()