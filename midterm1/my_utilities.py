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


def compute_err(actual, predictions):
    mae = np.mean(np.abs(np.subtract(predictions, actual)))
    mape = np.mean(np.abs(np.divide(np.subtract(actual, predictions), actual)))
    return mae, mape


def plot_curves(actual, predictions, title):
    plt.plot(actual, label="Test data")
    plt.plot(predictions, label="Predictions")
    plt.title(title)
    plt.legend()
    plt.show()


def train_models_parallel(ar_order, ma_order, arma_ar_order, arma_ma_order, tr_data, ts_data, err_thresh: list = None):
    ret_dict = Manager().dict()
    p_ar = Process(target=train_ar, args=(ar_order, copy.deepcopy(tr_data), ts_data, ret_dict, err_thresh))
    p_arma = Process(target=train_arma,
                     args=(arma_ar_order, arma_ma_order, copy.deepcopy(tr_data), ts_data, ret_dict, err_thresh))
    p_ar.start()
    p_arma.start()
    p_ar.join()
    p_arma.join()
    return ret_dict


def train_ar(order, tr_data, ts_data, ret_dict, err_thresh: list = None):
    predictions = []
    for i in tqdm(range(24 * 6)):
        ar = ARIMA(endog=tr_data, order=(order, 0, 0))
        res_ar = ar.fit()
        predictions.append(res_ar.forecast(steps=1))
        # curr_err = abs(predictions[-1] - ts_data[i])
        tr_data = np.concatenate((tr_data, [ts_data[i]]))

    mae, mape = compute_err(ts_data[:24 * 6], predictions)
    ret_dict["ar"] = {"mae": mae, "mape": mape}
    plot_curves(ts_data[:24 * 6], predictions, f"AR model (order: {order})")


def train_arma(ar_order, ma_order, tr_data, ts_data, ret_dict, err_thresh: list = None):
    predictions = []
    for i in tqdm(range(24 * 6)):
        arma = ARIMA(endog=tr_data, order=(ar_order, 0, ma_order))
        res_ar = arma.fit()
        predictions.append(res_ar.forecast(steps=1))
        # curr_err = abs(predictions[-1] - ts_data[i])
        tr_data = np.concatenate((tr_data, [ts_data[i]]))

    mae, mape = compute_err(ts_data[:24 * 6], predictions)
    ret_dict["arma"] = {"mae": mae, "mape": mape}
    plot_curves(ts_data[:24 * 6], predictions, f"ARMA model (AR order: {ar_order} - MA order: {ma_order})")


if __name__ == '__main__':
    ar_order = 3
    ma_order = 4
    arma_ar_order = 3
    arma_ma_order = 1
    _, tr_data, ts_data = read_data()
    metrics = train_models_parallel(ar_order, ma_order, arma_ar_order, arma_ma_order, tr_data[:500], ts_data)
    for k, v in metrics.items():
        print(f"{k}:")
        for kk, vv in metrics[k].items():
            print(f"{kk}: {vv}")
        print()
