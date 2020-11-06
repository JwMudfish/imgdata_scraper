import pandas_datareader.data as web

import datetime

start = datetime.datetime(2020, 11, 1)

end = datetime.datetime(2020, 11, 4)

gs = web.DataReader("078930.KS", "yahoo", start, end)
print(gs)