import pandas as pd
import datetime
import numpy as np
from statsmodels.tsa.stattools import levinson_durbin


# read the data
data = pd.read_csv("energydata_complete.csv")
data = data[["date", "Appliances"]]
data["date"] = pd.to_datetime(data["date"])
data = data.sort_values(by="date")

# select first week of data
first_day = data["date"][0]
last_day = first_day + datetime.timedelta(days=7)
first_week = data.loc[(data["date"] >= first_day) & (data["date"] <= last_day)]

err, alphas, pacf, sigma, phi = levinson_durbin(first_week["Appliances"], nlags=10, isacov=False)

print(np.shape(err))
print(np.shape(alphas))
print(np.shape(pacf))
print(np.shape(sigma))
print(np.shape(phi))

print(err)
print(alphas)
