import unittest
from RBM import RBM
from utilities import load_mnist


class MyTestCase(unittest.TestCase):
    def test_rbm(self):
        imgs, _, _, _ = load_mnist(path="../MNIST/")
        rbm = RBM(n_visible=len(imgs[0]))
        rbm.fit(epochs=1, lr=0.1, k=1, mnist_path="../MNIST/", fit_classifier=True, save_path="../models/prova.pickle")
        # rbm.save_model('here.pickle')
        # rbm.load_weights('here.pickle')


if __name__ == '__main__':
    unittest.main()
