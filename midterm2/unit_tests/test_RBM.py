import unittest
import matplotlib.pyplot as plt
import numpy as np
from RBM import RBM
from utilities import load_mnist


class MyTestCase(unittest.TestCase):
    def test_rbm(self):
        imgs, _, _, _ = load_mnist(path="../MNIST/")
        rbm = RBM(n_visible=len(imgs[0]))
        rbm.fit(epochs=1,
                lr=0.1,
                k=1,
                mnist_path="../MNIST/",
                fit_classifier=True,
                save_path="../models/rbm_weights.pickle",
                show_feats=True)

        # rbm.show_embedding(imgs[0])
        # rbm.show_embedding(imgs[1])
        # rbm.show_embedding(imgs[2])
        # rbm.show_embedding(imgs[3])
        # fig, ax = plt.subplots(2, 2)
        # ax[0, 0].imshow(np.reshape(imgs[0], newshape=(28, 28)))
        # ax[0, 1].imshow(np.reshape(imgs[1], newshape=(28, 28)))
        # ax[1, 0].imshow(np.reshape(imgs[2], newshape=(28, 28)))
        # ax[1, 1].imshow(np.reshape(imgs[3], newshape=(28, 28)))
        # fig.show()

        # rbm.save_model('here.pickle')
        # rbm.load_weights('here.pickle')
        # rbm.fit_classifier(load_rbm_weights=True, w_path='../models/prova.pickle', mnist_path='../MNIST/', save=False)
        # rbm.test_classifier(mnist_path='../MNIST/')


if __name__ == '__main__':
    unittest.main()
