import seaborn as sns
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import datetime

sns.set_style("darkgrid")

df = pd.read_csv("ibex_35.csv")

df = df.dropna()

log_returns = np.diff(np.log(df["Close"].values))
date = df["Date"].values[1:]
date = np.array([datetime.datetime.strptime(x , "%Y-%m-%d") for x in date])

years = mdates.YearLocator()
months = mdates.MonthLocator()
years_fmt = mdates.DateFormatter('%Y')


fig, ax = plt.subplots()
ax.plot(date, log_returns , 'k')
ax.set_ylabel("$R_{(t)}$")
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)
ax.set_title("Daily log returns IBEX-35")
datemin = np.datetime64(date[0], 'Y')
datemax = np.datetime64(date[-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)
idx = np.round(np.linspace(0, len(date)-1 , 8)).astype(int)
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.format_ydata = lambda x: '$%1.2f' % x
ax.set_xticks(date[idx])
fig.autofmt_xdate()
plt.savefig("ibexlogreturns.pdf")
