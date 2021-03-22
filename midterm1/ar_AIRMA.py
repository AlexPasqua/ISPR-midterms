import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import copy
from tqdm import tqdm
from my_utilities import read_data

if __name__ == '__main__':
    whole_data, tr_data, ts_data = read_data()

    # # plot autocorrelation of the whole training data
    # plot_acf(tr_data, lags=len(tr_data)-1)
    # plt.show()
    #
    # # plot autocorrelation and partial autocorrelation with limited lags
    # plot_acf(tr_data, lags=30)
    # plt.show()
    # plot_pacf(tr_data, lags=30)
    # plt.show()

    # PROVA
    # ts_data = list(tr_data[11563: 11563 + 10])
    # tr_data = list(tr_data[:11563])
    # predictions = []
    # model = ARIMA(endog=tr_data[:11563], order=(3, 0, 0))
    # res = model.fit()
    # print("Coefficients: ", res.arparams)
    # for i in tqdm(range(10)):
    #     predictions.append(res.forecast(steps=1))
    #     tr_data.append(ts_data[i])
    #     model = ARIMA(endog=tr_data, order=(3, 0, 0))
    #     res = model.fit()
    #
    # plt.plot(ts_data, label="Test data")
    # plt.plot(predictions, label="Prediction")
    # plt.legend()
    # plt.show()
    # exit()

    """ looks like an AR series with lags <= 15 """

    original_tr_data = copy.deepcopy(tr_data)
    retrain = True
    orders = [1, 2, 3, 5, 10, 15]
    for order in orders:
        # fit a model
        model = ARIMA(endog=original_tr_data, order=(order, 0, 0))
        res = model.fit()
        print("Coefficients: ", res.arparams)

        # start predicting and retraining
        err = 0
        predictions = []
        last_retrain_idx = 0
        count_no_retrain = 1
        for i in tqdm(range(len(ts_data))):
            predictions.append(res.forecast(steps=count_no_retrain)[-1])
            curr_err = abs(predictions[-1] - ts_data[i])
            err += curr_err
            count_no_retrain += 1
            if retrain and i % 20 == 0:
                tr_data = np.concatenate((tr_data, ts_data[last_retrain_idx: i]))
                last_retrain_idx = i
                count_no_retrain = 1
                model = ARIMA(endog=tr_data, order=(order, 0, 0))
                res = model.fit()

        err /= len(ts_data)
        print("Average error: ", err)
        plt.plot(ts_data, label="Test data")
        plt.plot(predictions, label="Predictions")
        plt.legend()
        plt.show()


    # retrain = True
    # count_no_retrain = 1
    # predictions = []
    # err = last_retrain_idx = 0
    # for i in range(len(ts_data)):
    #     predictions.append(res.forecast(steps=count_no_retrain)[-1])
    #     prev_err = abs(predictions[-1] - ts_data[i])
    #     err += prev_err
    #     tr_data.append(ts_data[i])
    #     if retrain and prev_err > 200:
    #         print(i)
    #         count_no_retrain = 1
    #         # tr_data.append(ts_data[last_retrain_idx: i + 1])
    #         last_retrain_idx = i
    #         model = ARIMA(endog=tr_data, order=(order, 0, 0))
    #         res = model.fit()
    #     else:
    #         count_no_retrain += 1


