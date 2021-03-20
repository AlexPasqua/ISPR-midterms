import pandas as pd
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
import datetime


def read_data():
    # read the data
    whole_data = pd.read_csv("energydata_complete.csv")
    whole_data = whole_data[["date", "Appliances"]]
    whole_data["date"] = pd.to_datetime(whole_data["date"])
    whole_data = whole_data.sort_values(by="date")

    # select training data
    first_day = whole_data["date"][0]
    last_day = first_day + datetime.timedelta(weeks=13)
    tr_data = whole_data[(whole_data["date"] >= first_day) & (whole_data["date"] <= last_day)]["Appliances"]
    test_data = whole_data[whole_data["date"] > last_day]["Appliances"]
    whole_data = whole_data["Appliances"]

    # autocorrelation_plot(tr_data)
    # plt.show()

    return whole_data.to_list(), tr_data.to_list(), test_data.to_list()
