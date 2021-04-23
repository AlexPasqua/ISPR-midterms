import unittest
from DRBN import DRBN
from utilities import load_mnist


class TestDRBN(unittest.TestCase):
    def test_drbn(self):
        tr_imgs, tr_labels, ts_imgs, ts_labels = load_mnist('../MNIST/')
        drbn = DRBN(hl_sizes=(100, 100), v_size=len(tr_imgs[0]), mnist_path='../MNIST/')
        drbn.fit(epochs=1,
                 lr=0.05,
                 k=1,
                 bs=1)
        drbn.show_reconstruction(img=tr_imgs[0])
        drbn.show_reconstruction(img=tr_imgs[1])
        drbn.show_reconstruction(img=tr_imgs[2])
        drbn.show_reconstruction(img=tr_imgs[3])


if __name__ == '__main__':
    unittest.main()
