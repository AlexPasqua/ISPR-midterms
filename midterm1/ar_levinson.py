import numpy as np
from statsmodels.tsa.stattools import levinson_durbin
from my_utilities import read_data

if __name__ == '__main__':
    # read the data
    whole_data, tr_data, ts_data = read_data()

    order = 4
    sigma_v, arcoefs, pacf, sigma, phi = levinson_durbin(tr_data, nlags=order, isacov=False)

    # print(np.shape(sigma_v))
    # print(arcoefs)
    # print(np.shape(pacf))
    # print(np.shape(sigma))
    # print(np.shape(phi))
    # print(sigma_v)
    # print(arcoefs)

    # prev_pts = tr_data[-(order + 1): -1]
    # predicted = np.dot(arcoefs[::-1], prev_pts)
    # print(predicted, tr_data[-1])
