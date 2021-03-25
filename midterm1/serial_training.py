import argparse
import json
import numpy as np
from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA
from my_utilities import read_data


def train(order_ar, order_ma, tr_data, ts_data, retrain, err_thresh):
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
            tr_data = np.concatenate((tr_data, ts_data[idx_last_retrain: i+1]))
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--order_ar', action='store', type=int, help="The AR order")
    parser.add_argument('--order_ma', action='store', type=int, help="The MA order")
    parser.add_argument('--err_thresh', action='store', type=int, help="Error threshold over which the model is retrained")
    parser.add_argument('--retrain', action='store_true', help="Whether or not to retrain during testing")
    args = parser.parse_args()

    _, tr_data, ts_data = read_data()
    train(order_ar=args.order_ar, order_ma=args.order_ma, tr_data=tr_data, ts_data=ts_data, retrain=args.retrain,
          err_thresh=args.err_thresh)
