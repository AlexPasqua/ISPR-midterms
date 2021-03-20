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
    # print(res.summary())
    # print(res.polynomial_ar)

    n_ts_samples = len(ts_data)
    predictions = [res.forecast(steps=1)]
    prev_err = err = abs(predictions[-1] - ts_data[0])
    for i in range(n_ts_samples - 1):
        tr_data.append(ts_data[i])
        # ts_data.remove(ts_data[i])
        if i % 20 == 0:  # prev_err > 65:
            print(i)
            model = ARIMA(endog=tr_data, order=(order, 0, 0))
            res = model.fit()
        predictions.append(res.forecast(steps=1))
        prev_err = abs(ts_data[i + 1] - predictions[-1])
        err += prev_err

    err /= n_ts_samples
    print(err)
    plt.plot(predictions)
    plt.show()
    plt.plot(ts_data)
    plt.show()
