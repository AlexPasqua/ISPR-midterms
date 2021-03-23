import datetime
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Manager
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm


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


def train_models(ar_order, ma_order, arma_ar_order, arma_ma_order, tr_data, ts_data):
    ar = ARIMA(endog=tr_data, order=(ar_order, 0, 0))
    ma = ARIMA(endog=tr_data, order=(0, 0, ma_order))
    arma = ARIMA(endog=tr_data, order=(arma_ar_order, 0, arma_ma_order))
    ret_dict = Manager().dict()
    p_ar = Process(target=train_ar, args=(ar_order, copy.deepcopy(tr_data), ts_data, ret_dict))
    p_ma = Process(target=train_ma, args=(ma_order, copy.deepcopy(tr_data), ts_data, ret_dict))
    p_arma = Process(target=train_arma, args=(arma_ar_order, arma_ma_order, copy.deepcopy(tr_data), ts_data, ret_dict))
    p_ar.start()
    p_ma.start()
    p_arma.start()
    p_ar.join()
    p_ma.join()
    p_arma.join()
    return ret_dict


def train_ar(order, tr_data, ts_data, ret_dict):
    predictions = []
    for i in tqdm(range(24*6)):
        ar = ARIMA(endog=tr_data, order=(order, 0, 0))
        res_ar = ar.fit()
        predictions.append(res_ar.forecast(steps=1))
        # curr_err = abs(predictions[-1] - ts_data[i])
        tr_data = np.concatenate((tr_data, [ts_data[i]]))

    mee = np.mean(np.abs(np.subtract(predictions, ts_data[:24*6])))
    mape = np.mean(np.divide(
        np.abs(np.subtract(predictions, ts_data[:24*6])),
        np.abs(ts_data[:24*6])
    ))
    ret_dict["ar"] = {"mee": mee, "mape": mape}
    plt.plot(ts_data[:24*6], label="Test data")
    plt.plot(predictions, label="Predictions")
    plt.title(f"AR model (order: {order})")
    plt.legend()
    plt.show()


def train_ma(order, tr_data, ts_data, ret_dict):
    predictions = []
    for i in tqdm(range(24 * 6)):
        ma = ARIMA(endog=tr_data, order=(0, 0, order))
        res_ar = ma.fit()
        predictions.append(res_ar.forecast(steps=1))
        # curr_err = abs(predictions[-1] - ts_data[i])
        tr_data = np.concatenate((tr_data, [ts_data[i]]))

    mee = np.mean(np.abs(np.subtract(predictions, ts_data[:24 * 6])))
    mape = np.mean(np.divide(
        np.abs(np.subtract(predictions, ts_data[:24 * 6])),
        np.abs(ts_data[:24 * 6])
    ))
    ret_dict["ma"] = {"mee": mee, "mape": mape}
    plt.plot(ts_data[:24 * 6], label="Test data")
    plt.plot(predictions, label="Predictions")
    plt.title(f"MA model (order: {order})")
    plt.legend()
    plt.show()


def train_arma(ar_order, ma_order, tr_data, ts_data, ret_dict):
    predictions = []
    for i in tqdm(range(24 * 6)):
        arma = ARIMA(endog=tr_data, order=(ar_order, 0, ma_order))
        res_ar = arma.fit()
        predictions.append(res_ar.forecast(steps=1))
        # curr_err = abs(predictions[-1] - ts_data[i])
        tr_data = np.concatenate((tr_data, [ts_data[i]]))

    mee = np.mean(np.abs(np.subtract(predictions, ts_data[:24 * 6])))
    mape = np.mean(np.divide(
        np.abs(np.subtract(predictions, ts_data[:24 * 6])),
        np.abs(ts_data[:24 * 6])
    ))
    ret_dict["arma"] = {"mee": mee, "mape": mape}
    plt.plot(ts_data[:24 * 6], label="Test data")
    plt.plot(predictions, label="Predictions")
    plt.title(f"ARMA model (AR order: {ar_order} - MA order: {ma_order})")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    ar_order = 3
    ma_order = 4
    arma_ar_order = 3
    arma_ma_order = 1
    _, tr_data, ts_data = read_data()
    metrics = train_models(ar_order, ma_order, arma_ar_order, arma_ma_order, tr_data[:500], ts_data)
    for k, v in metrics.items():
        print(f"{k}:")
        for kk, vv in metrics[k].items():
            print(f"{kk}: {vv}")
        print()