import unittest
from DRBN import DRBN
from utilities import load_mnist


class TestDRBN(unittest.TestCase):
    def test_drbn(self):
        tr_imgs, tr_labels, ts_imgs, ts_labels = load_mnist('../MNIST/')
        # drbn = DRBN(hl_sizes=(500, 100), v_size=len(tr_imgs[0]), mnist_path='../MNIST/')
        # drbn.fit(epochs=1,
        #          lr=0.05,
        #          k=1,
        #          bs=10,
        #          save=False,
        #          save_path=None,
        #          fit_cl=True,
        #          save_cl=False,
        #          save_cl_path=None)
        # drbn.show_reconstruction(img=tr_imgs[0])
        # drbn.show_reconstruction(img=tr_imgs[1])
        # drbn.test_classifier()
        # drbn.show_reconstruction(img=tr_imgs[2])
        # drbn.show_reconstruction(img=tr_imgs[3])
        # drbn.save_model('../models/DRBN_weights.pickle')
        new_drbn = DRBN(hl_sizes=(500, 100), v_size=len(tr_imgs[0]), mnist_path='../MNIST/')
        new_drbn.load_weights('../models/DRBN_weights.pickle')
        new_drbn.fit_classifier()
        new_drbn.test_classifier()
        new_drbn.show_reconstruction(img=tr_imgs[0])
        new_drbn.show_reconstruction(img=tr_imgs[1])


if __name__ == '__main__':
    unittest.main()
