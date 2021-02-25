import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

sns.set_style("darkgrid")

df = pd.read_csv("ibex_35.csv")

df = df.dropna()

log_returns = np.diff(np.log(df["Close"].values))

mu , std = norm.fit(log_returns)

fig , ax = plt.subplots(figsize=(12,8))
plt.rcParams.update({'font.size': 16})
ret = plt.hist(log_returns , bins = 100 , density = True , color="dodgerblue" , label = "Empirical Distribution")
xmin , xmax = plt.xlim()

x = np.linspace(xmin , xmax , 100)
p = norm.pdf(x , mu , std)
plt.plot(x , p ,'k' , linewidth=2 , label = "Gaussian Fit")
plt.legend(loc=2)

plt.title("IBEX-35 Log Returns Distribution")
plt.xlabel("$R_{(t)}$")


axins1 = zoomed_inset_axes(ax, zoom = 5, loc=1)
axins1.hist(log_returns , bins=100 , density = True , color="dodgerblue" , label="Fat Tails")
axins1.plot(x,p , 'k')
axins1.legend(loc=1)
x1, x2, y1, y2 = 0.035,0.060,0.05,2 
axins1.set_xlim(x1, x2)
axins1.set_ylim(y1, y2)
plt.yticks(visible=False)
mark_inset(ax, axins1, loc1=4, loc2=3, fc="none", ec="0.5")
plt.savefig("gaussian_distributionibex.pdf")
