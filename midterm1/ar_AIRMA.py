import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from my_utilities import read_data

if __name__ == '__main__':
    whole_data, tr_data = read_data()
    # plt.plot(tr_data)
    # plt.show()
    order = 4
    model = ARIMA(endog=tr_data, order=(order, 0, 0))
    res = model.fit()
    print(res.summary())
    print(res.polynomial_ar)
    print(res.predict(start=len(tr_data)-1))
    print(tr_data[len(tr_data)-1])
