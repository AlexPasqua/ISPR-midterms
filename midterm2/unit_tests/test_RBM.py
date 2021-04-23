import unittest
import matplotlib.pyplot as plt
import numpy as np
from RBM import RBM
import tensorflow as tf
from utilities import load_mnist


class MyTestCase(unittest.TestCase):
    def test_rbm(self):
        imgs, _, _, _ = load_mnist(path="../MNIST/")
        rbm = RBM(n_visible=len(imgs[0]), mnist_path='../MNIST/')
        rbm.fit(epochs=1,
                lr=0.1,
                k=1,
                bs='batch',
                save=False,
                save_path="../models/rbm_weights.pickle",
                fit_cl=True,
                save_cl=False,
                save_cl_path=None,
                show_feats=True)

        # rbm.show_encoding(imgs[0])
        # rbm.show_encoding(imgs[1])
        # rbm.show_encoding(imgs[2])
        # rbm.show_encoding(imgs[3])
        # fig, ax = plt.subplots(2, 2)
        # ax[0, 0].imshow(np.reshape(imgs[0], newshape=(28, 28)))
        # ax[0, 1].imshow(np.reshape(imgs[1], newshape=(28, 28)))
        # ax[1, 0].imshow(np.reshape(imgs[2], newshape=(28, 28)))
        # ax[1, 1].imshow(np.reshape(imgs[3], newshape=(28, 28)))
        # fig.show()

        # rbm.save_model('here.pickle')
        # rbm.load_weights('here.pickle')
        # rbm.fit_classifier(load_rbm_weights=True, w_path='../models/rbm_weights.pickle', save=True)
        rbm.test_classifier()
        rbm.show_reconstruction(imgs[0])


if __name__ == '__main__':
    unittest.main()
