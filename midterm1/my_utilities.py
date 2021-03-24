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
    # mape = np.mean(np.abs(np.divide(np.subtract(actual, predictions), actual)))
    return mae


def plot_curves(actual, predictions, title):
    plt.plot(actual, label="Test data")
    plt.plot(predictions, label="Predictions")
    plt.title(title)
    plt.legend()
    plt.show()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def train_models_parallel(ar_order, arma_ar_order, arma_ma_order, tr_data, ts_data, err_threshes):
    ret_dict = Manager().dict()
    procs = []
    idx = 0
    for i in range(len(err_threshes) * 2):
        idx = idx + 1 if i // (idx + 1) >= 2 else idx
        err_thresh = err_threshes[idx]
        if i % 2 == 0:
            procs.append(Process(target=train_ar, args=(ar_order, copy.deepcopy(tr_data), ts_data, ret_dict, err_thresh)))
        else:
            procs.append(Process(target=train_arma, args=(arma_ar_order, arma_ma_order, copy.deepcopy(tr_data), ts_data,
                                                          ret_dict, err_thresh)))
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    return ret_dict


def train_ar(order, tr_data, ts_data, ret_dict, err_thresh):
    predictions = []
    last_retrain_idx = 0
    count_no_retrain = 1
    err = 1000000
    for i in tqdm(range(len(ts_data))):
        if i == 0 or err > err_thresh:
            last_retrain_idx = i
            count_no_retrain = 0
            tr_data = np.concatenate((tr_data, ts_data[last_retrain_idx: i]))
            ar = ARIMA(endog=tr_data, order=(order, 0, 0))
            res_ar = ar.fit()
        count_no_retrain += 1
        predictions.append(res_ar.forecast(steps=count_no_retrain)[-1])
        err = abs(predictions[-1] - ts_data[i])
        # tr_data = np.concatenate((tr_data, [ts_data[i]]))

    mae = compute_err(ts_data, predictions)
    ret_dict["ar_" + str(err_thresh)] = mae
    # plot the last 2 days
    plot_curves(ts_data[-(48 * 6):], predictions[-(48 * 6):], f"AR model (order: {order})")


def train_arma(ar_order, ma_order, tr_data, ts_data, ret_dict, err_thresh):
    predictions = []
    last_retrain_idx = 0
    count_no_retrain = 1
    err = 1000000
    for i in tqdm(range(len(ts_data))):
        if i == 0 or err > err_thresh:
            last_retrain_idx = i
            count_no_retrain = 0
            tr_data = np.concatenate((tr_data, ts_data[last_retrain_idx: i]))
            arma = ARIMA(endog=tr_data, order=(ar_order, 0, ma_order))
            res_ar = arma.fit()
        count_no_retrain += 1
        predictions.append(res_ar.forecast(steps=count_no_retrain)[-1])
        err = abs(predictions[-1] - ts_data[i])
        # tr_data = np.concatenate((tr_data, [ts_data[i]]))

    mae = compute_err(ts_data, predictions)
    ret_dict["arma_" + str(err_thresh)] = mae
    # plot the last 2 days
    plot_curves(ts_data[-(48 * 6):], predictions[-(48 * 6):], f"ARMA model (AR order: {ar_order} - MA order: {ma_order})")


if __name__ == '__main__':
    ar_order = 3
    arma_ar_order = 3
    arma_ma_order = 1
    err_threses = [10, 50, 100]
    _, tr_data, ts_data = read_data()
    metrics = train_models_parallel(ar_order, arma_ar_order, arma_ma_order, tr_data, ts_data[:500], err_threses)

    # print error values
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # plot histogram
    labels = [f"AR({ar_order})", f"ARMA({arma_ar_order}, {arma_ma_order})"]
    series1 = [round(metrics["ar_" + str(err_threses[0])]), round(metrics["arma_" + str(err_threses[0])])]
    series2 = [round(metrics["ar_" + str(err_threses[1])]), round(metrics["arma_" + str(err_threses[1])])]
    series3 = [round(metrics["ar_" + str(err_threses[2])]), round(metrics["arma_" + str(err_threses[2])])]
    x = np.arange(len(labels))
    bars_width = 0.2
    fig, ax = plt.subplots()
    r1 = ax.bar(x - bars_width, series1, bars_width, label="Error threshold: " + str(err_threses[0]))
    r2 = ax.bar(x, series2, bars_width, label="Error threshold: " + str(err_threses[1]))
    r3 = ax.bar(x + bars_width, series3, bars_width, label="Error threshold: " + str(err_threses[2]))
    ax.set_ylabel("MAE")
    ax.set_title("Mean Absolute Error")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    autolabel(r1)
    autolabel(r2)
    autolabel(r3)
    fig.tight_layout()
    plt.show()
