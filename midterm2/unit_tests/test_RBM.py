import unittest
from RBM import RBM
from utilities import load_mnist


class MyTestCase(unittest.TestCase):
    def test_gibbs_sampling(self):
        imgs, _, _, _ = load_mnist(path="../MNIST/")
        rbm = RBM(n_visible=len(imgs[0]))
        rbm.fit(epochs=1, lr=0.1, k=1, mnist_path="../MNIST/")


if __name__ == '__main__':
    unittest.main()
