import pandas as pd
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
import datetime


def train_models(ar, ma, arma):
    """
    Functions to train the models in parallel
    """
    pass


def train_ar():
    pass


def train_ma():
    pass


def train_arma():
    pass


def read_data():
    """
    Reads the data from energydata_complete.csv
    :return: the whole "Appliances" column, the first 3 months of it (as tr_data) and the last 1.5 months (ts_data)
    """
    # read the data and sort it by date
    whole_data = pd.read_csv("energydata_complete.csv")
    whole_data = whole_data[["date", "Appliances"]]
    whole_data["date"] = pd.to_datetime(whole_data["date"])
    whole_data = whole_data.sort_values(by="date")

    # select training and test data. Then drop the date column
    first_day = whole_data["date"][0]
    last_day = first_day + datetime.timedelta(weeks=13)
    tr_data = whole_data[(whole_data["date"] >= first_day) & (whole_data["date"] <= last_day)]["Appliances"]
    test_data = whole_data[whole_data["date"] > last_day]["Appliances"]
    whole_data = whole_data["Appliances"]

    return whole_data.to_numpy(), tr_data.to_numpy(), test_data.to_numpy()
