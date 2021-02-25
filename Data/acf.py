from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


sns.set_style("darkgrid")

df = pd.read_csv("ibex_35.csv")

df = df.dropna()

log_returns = np.diff(np.log(df["Close"].values))
plot_acf(log_returns , lags=30)
plt.title("IBEX-35 Daily log returns autocorrelation")
plt.xlabel("Lag")
plt.ylabel("ACF")
plt.savefig("acfibex.pdf")
plt.show()
