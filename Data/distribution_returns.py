import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

sns.set_style("darkgrid")

df = pd.read_csv("ibex_35.csv")

df = df.dropna()

log_returns = np.diff(np.log(df["Close"].values))

#new_df = pd.Dataframe(data = log_returns , columns = ["Log Returns"])
ax = sns.distplot(log_returns , fit=norm , kde=False , hist_kws={"color":"dodgerblue" , "label":"Empirical Distribution"} , fit_kws={"color":"k" , "label":"Gaussian Fit"})
plt.title("IBEX-35 Log Returns Distribution")
plt.xlabel("$R_{(t)}$")
plt.legend(loc=1)
plt.savefig("gaussian_distributionibex.pdf")

