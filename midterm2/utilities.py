import pickle
import numpy as np
from mlxtend.data import loadlocal_mnist
from RBM import RBM


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


def load_model(path):
    """
    Loads a model from pickle file
    :param path: path to the pickle file containing the model
    :return: the RBM object
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return RBM(
        n_visible=data['n_visible'],
        n_hidden=data['n_hidden'],
        W=data['weights'],
        bias_visible=data['bias_visible'],
        bias_hidden=data['bias_hidden']
    )


def sigmoid(x):
    """ Computes the sigmoid function """
    ones = [1.] * len(x)
    return np.divide(ones, np.add(ones, np.exp(-x)))


if __name__ == '__main__':
    tr_imgs, tr_labels, ts_imgs, ts_labels = load_mnist()
    print('Dimensions: %s x %s' % (tr_imgs.shape[0], tr_imgs.shape[1]))

    # img = np.reshape(train_images[0], newshape=(28, 28))
    # plt.imshow(img)
    # plt.show()
