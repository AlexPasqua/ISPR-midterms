import numpy as np
from mlxtend.data import loadlocal_mnist


def load_mnist(path=None):
    """
    Load the MNIST dataset
    :param path: path to the directory containing the MNIST files
    :return: training and test images and labels scaled between 0 and 1
    """
    path = "MNIST/" if path is None else (path + '/' if path[-1] != '/' else path)
    train_images, train_labels = loadlocal_mnist(images_path=path + 'train-images-idx3-ubyte',
                                                 labels_path=path + 'train-labels-idx1-ubyte')
    test_images, test_labels = loadlocal_mnist(images_path=path + 't10k-images-idx3-ubyte',
                                               labels_path=path + 't10k-labels-idx1-ubyte')
    # 0-1 scaling
    min_tr, max_tr = np.min(train_images), np.max(train_images)
    min_ts, max_ts = np.min(test_images), np.max(test_images)
    train_images = np.divide(np.subtract(train_images, min_tr), max_tr - min_tr)
    test_images = np.divide(np.subtract(test_images, min_ts), max_ts - min_ts)
    return train_images, train_labels, test_images, test_labels


def sigmoid(x):
    """ Computes the sigmoid function """
    ones = [1.] * len(x)
    return np.divide(ones, np.add(ones, np.exp(-x)))

