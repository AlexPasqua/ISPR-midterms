import datetime
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
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


def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def my_bar_plot(series, labels):
    """
    Plots the bar-plot of the MAE for each model, for each retraining schedule
    :param series: list of series to plot (one for each model, one value for each retraining schedule (err_thresh))
    :param labels: labels for the xtickslabels
    """
    series_list = list(series.values())
    series0, series1, series2, series3 = series_list[0], series_list[1], series_list[2], series_list[3]
    x = np.arange(len(labels))
    bars_width = 0.22
    fig, ax = plt.subplots(figsize=(10, 7))
    r0 = ax.bar(x - 1.5 * bars_width, series0, bars_width, label="Error threshold: " + str(list(series.keys())[0]))
    r1 = ax.bar(x - 0.5 * bars_width, series1, bars_width, label="Error threshold: " + str(list(series.keys())[1]))
    r2 = ax.bar(x + 0.5 * bars_width, series2, bars_width, label="Error threshold: " + str(list(series.keys())[2]))
    r3 = ax.bar(x + 1.5 * bars_width, series3, bars_width, label="Error threshold: " + str(list(series.keys())[3]))
    ax.set_ylabel("MAE")
    ax.set_title("Mean Absolute Error")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    autolabel(ax, r0)
    autolabel(ax, r1)
    autolabel(ax, r2)
    autolabel(ax, r3)
    # fig.tight_layout()
    plt.show()


def predict_and_retrain(order_ar, order_ma, tr_data, ts_data, retrain, err_thresh):
    """
    Perform a cycle of predicting and retraining
    :param order_ar: order of the autoregressive part
    :param order_ma: order of the moving average part
    :param tr_data: training data -> got from function read_data()
    :param ts_data: test data -> got from function read_data()
    :param retrain: boolean, if False the model is never retrained
    :param err_thresh: the error threshold over which a retraining is performed
    """
    # create and fit the model on the training set
    model = ARIMA(endog=tr_data, order=(order_ar, 0, order_ma))
    res = model.fit()

    # start forecasting on the test set and retrain when needed
    idx_last_retrain = 0
    count_no_retrain = 1
    predictions = []
    for i in tqdm(range(len(ts_data))):
        predictions.append(res.forecast(steps=count_no_retrain)[-1])
        err = abs(ts_data[i] - predictions[-1])
        if retrain and err > err_thresh:
            idx_last_retrain = i
            count_no_retrain = 1
            tr_data = np.concatenate((tr_data, ts_data[idx_last_retrain: i + 1]))
            model = ARIMA(endog=tr_data, order=(order_ar, 0, order_ma))
            res = model.fit()
        else:
            count_no_retrain += 1

    # compute mean absolute error (MAE)
    mae = np.mean(np.abs(np.subtract(ts_data, predictions)))
    out = {'mae': mae, 'predictions': predictions}
    filename = str(order_ar) + "_" + str(order_ma) + "_" + str(err_thresh) + "_retrain" if retrain else "" + ".json"
    with open(filename, 'w') as outf:
        json.dump(out, outf, indent='\t')


def join_files(path):
    """
    Function to take all the files of the results and create one single file with the results of every model
    :param path: path of the directory where the results files are
    """
    results = []
    filenames = [f for f in listdir(path) if isfile(join(path, f))]
    for fn in filenames:
        if fn[-5:] == ".json":
            with open(fn, 'r') as f:
                data = json.load(f)
                data = {**{"model": fn[:-5]}, **data}
                results.append(data)

    with open("results/whole_results.json", 'w') as outf:
        json.dump(results, outf, indent='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--order_ar', action='store', type=int, help="The AR order")
    parser.add_argument('--order_ma', action='store', type=int, help="The MA order")
    parser.add_argument('--err_thresh', action='store', type=int, help="Error threshold over which the model is retrained")
    parser.add_argument('--retrain', action='store_true', help="Whether or not to retrain during testing")
    args = parser.parse_args()

    _, tr_data, ts_data = read_data()
    predict_and_retrain(order_ar=args.order_ar, order_ma=args.order_ma, tr_data=tr_data, ts_data=ts_data, retrain=args.retrain,
                        err_thresh=args.err_thresh)
