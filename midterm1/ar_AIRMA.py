import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from my_utilities import read_data

if __name__ == '__main__':
    whole_data, tr_data, ts_data = read_data()

    # plot autocorrelation and partial autocorrelation
    # plot_acf(tr_data, lags=20)
    # plt.show()
    # plot_pacf(tr_data, lags=20)
    # plt.show()

    order = 4
    model = ARIMA(endog=tr_data, order=(order, 0, 0))
    res = model.fit()
    print(res.summary())
    # print(res.polynomial_ar)


    retrain = True
    count_no_retrain = 1
    n_ts_samples = len(ts_data)
    predictions = []
    err = last_retrain_idx = 0
    for i in range(n_ts_samples):
        tr_data.append(ts_data[i])
        predictions.append(res.forecast(steps=count_no_retrain)[-1])
        prev_err = abs(ts_data[i] - predictions[-1])
        err += prev_err
        if retrain and prev_err > 200:
            print(i)
            count_no_retrain = 1
            # tr_data.append(ts_data[last_retrain_idx: i + 1])
            last_retrain_idx = i
            model = ARIMA(endog=tr_data, order=(order, 0, 0))
            res = model.fit()
        else:
            count_no_retrain += 1

    err /= n_ts_samples
    print(err)
    plt.plot(predictions)
    plt.show()
    plt.plot(ts_data)
    plt.show()
