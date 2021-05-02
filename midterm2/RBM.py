import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from utilities import sigmoid, load_mnist
from DRBN import DRBN


class RBM(DRBN):
    """ Implementation of a Restricted Boltzmann Machine """

    def __init__(self, n_visible, n_hidden=100, W=None, bias_visible=None, bias_hidden=None, mnist_path=None):
        """
        Constructor
        :param n_visible: number of visible units
        :param n_hidden: number of hidden units
        :param W: weights matrix
        :param bias_visible: bias vector for visible layer
        :param bias_hidden: bias vector for hidden layer
        :param mnist_path: the path to the directory containing MNIST
        :raises AssertionError: if there are incompatibilities between parameters
        """
        super().__init__(hl_sizes=(n_hidden,), v_size=n_visible, mnist_path=mnist_path)
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        weights_shape = (n_hidden, n_visible)
        self.W_matrs[0] = W if W is not None else self.W_matrs[0]
        self.bias_visible = bias_visible if bias_visible is not None else self.bias_visible
        self.bias_hidden = bias_hidden if bias_hidden is not None else self.bias_hidden
        assert self.W_matrs[0].shape == weights_shape and \
               n_visible == len(self.bias_visible) and \
               n_hidden == len(self.bias_hidden)

    @property
    def W(self):
        return self.W_matrs[0]

    @W.setter
    def W(self, value):
        self.W_matrs[0] = value

    @property
    def bias_visible(self):
        return self.biases[0]

    @bias_visible.setter
    def bias_visible(self, value):
        self.biases[0] = value

    @property
    def bias_hidden(self):
        return self.biases[1]

    @bias_hidden.setter
    def bias_hidden(self, value):
        self.biases[1] = value

    def ph_v(self, v_sample):
        """ Compute conditional probability and samples of the hidden units given the visible ones """
        h_probs = sigmoid(np.add(np.matmul(self.W, v_sample), self.bias_hidden))
        h_samples = np.random.binomial(n=1, p=h_probs, size=len(h_probs))
        return h_probs, h_samples

    def pv_h(self, h_sample):
        """ Compute conditional probability and samples of the visible units given the hidden ones """
        v_probs = sigmoid(np.add(np.matmul(h_sample, self.W), self.bias_visible))
        v_samples = np.random.binomial(n=1, p=v_probs, size=len(v_probs))
        return v_probs, v_samples

    def gibbs_sampling(self, h_sample, k):
        """
        Performs Gibbs sampling
        :param h_sample: binary vector of the hidden activations
        :param k: order of the Gibbs sampling
        """
        v_prob, v_sample, h_prob = None, None, None
        for i in range(k):
            v_prob, v_sample = self.pv_h(h_sample)
            h_prob, h_sample = self.ph_v(v_sample)
        return v_prob, v_sample, h_prob, h_sample

    def contrastive_divergence(self, v_probs, k):
        """
        Perform one step of contrastive divergence
        :param v_probs: vector of the visible units activations (non-binarized)
        :param k: (int) order of the Gibbs sampling
        """
        # compute wake part
        v_sample = np.random.binomial(n=1, p=v_probs, size=len(v_probs))
        h_probs, h_sample = self.ph_v(v_sample)
        wake = np.outer(h_probs, v_probs)
        # compute dream part
        v_probs_gibbs, v_sample_gibbs, h_probs_gibbs, h_sample_gibbs = self.gibbs_sampling(h_sample, k)
        dream = np.outer(h_probs_gibbs, v_probs_gibbs)
        # compute deltas
        delta_W = np.subtract(wake, dream)
        delta_bv = np.subtract(v_sample, v_sample_gibbs)
        delta_bh = np.subtract(h_sample, h_sample_gibbs)
        return delta_W, delta_bv, delta_bh

    def fit(self, epochs=1, lr=0.1, k=1, bs=1, save=False, save_path=None, fit_cl=False, save_cl=False,
            save_cl_path=None, show_feats=False):
        """
        Perform model fitting by contrastive divergence
        :param epochs: (int) number of epochs of training
        :param lr: (float) learning rate (0 < lr <= 1)
        :param k: (int) order of the Gibbs sampling
        :param bs: (int) size of a batch/minibatch of training data
        :param save: (bool) if True, save the model's weights and biases
        :param save_path: (str) path where to save the model
        :param fit_cl: (bool) if True, fit the classifier on the encodings
        :param save_cl: (book) if True, save the classifier's weights
        :param save_cl_path: (str) path where to save the classifier
        :param show_feats: (bool) if True, plot the learnt features (reshaped rows of weights matrix)
        """
        assert epochs > 0 and 0 < lr <= 1 and k > 0
        # iterate over epochs
        n_imgs = len(self.tr_labels)
        bs = n_imgs if bs == 'batch' or bs > n_imgs else bs
        disable_tqdm = (False, True) if bs < n_imgs else (True, False)  # for progress bars
        indexes = list(range(len(self.tr_imgs)))
        for ep in range(epochs):
            # shuffle the data
            np.random.shuffle(indexes)
            self.tr_imgs = self.tr_imgs[indexes]
            self.tr_labels = self.tr_labels[indexes]
            # cycle through batches
            for batch_idx in tqdm(range(math.ceil(len(self.tr_labels) / bs)), disable=disable_tqdm[0]):
                delta_W = np.zeros(shape=self.W.shape)
                delta_bv = np.zeros(shape=(len(self.bias_visible),))
                delta_bh = np.zeros(shape=(len(self.bias_hidden),))
                start = batch_idx * bs
                end = start + bs
                batch_imgs = self.tr_imgs[start: end]
                # cycle through patterns within a batch
                for img in tqdm(batch_imgs, disable=disable_tqdm[1]):
                    dW, dbv, dbh = self.contrastive_divergence(v_probs=img, k=k)
                    delta_W = np.add(delta_W, dW)
                    delta_bv = np.add(delta_bv, dbv)
                    delta_bh = np.add(delta_bh, dbh)
                # weight update
                self.W += (lr / bs) * delta_W
                self.bias_visible += (lr / bs) * delta_bv
                self.bias_hidden += (lr / bs) * delta_bh
        if save:
            self.save_model(datetime.now().strftime("rbm_%d-%m-%y_%H-%M") if save_path is None else save_path)
        if show_feats:
            self.show_learnt_features()
        if fit_cl:
            # train the classifier on the embeddings
            self.fit_classifier(save=save_cl, save_path=save_cl_path)

    def show_learnt_features(self):
        """
        Shows the features learnt by the model,
        i.e. each row of the weights matrix reshaped as image
        """
        fig, ax = plt.subplots(10, 10, figsize=(8, 8))
        for i in range(10):
            for j in range(10):
                ax[i, j].imshow(np.reshape(self.W[10 * i + j], newshape=(28, 28)))
                ax[i, j].axis('off')
        fig.suptitle('Learnt features', fontsize='x-large')
        fig.tight_layout()
        fig.show()

    def show_reconstruction(self, img):
        """
        Plots a reconstruction of an image
        :param img: (vector) the image to reconstruct
        """
        v_sample = np.random.binomial(n=1, p=img, size=len(img))
        h_probs, h_sample = self.ph_v(v_sample)
        v_probs, v_sample = self.pv_h(h_sample)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(np.reshape(img, newshape=(28, 28)))
        ax[0].set_title('Original image')
        ax[1].imshow(np.reshape(v_probs, newshape=(28, 28)))
        ax[1].set_title('Reconstructed image')
        fig.suptitle('Reconstruction')
        fig.tight_layout()
        fig.show()

    def show_encoding(self, img=None):
        """
        Shows the embedding for one image
        :param img: vector representing an image
        """
        img = np.random.binomial(n=1, p=img, size=len(img))
        probs, samples = self.ph_v(img)
        fig, ax = plt.subplots(1, 2)
        side_len = int(np.sqrt(self.n_hidden))
        ax[0].imshow(np.reshape(probs, newshape=(side_len, side_len)), cmap='gray')
        ax[0].set_title('Probabilities')
        ax[1].imshow(np.reshape(samples, newshape=(side_len, side_len)), cmap='gray')
        ax[1].set_title('Samples')
        fig.suptitle('Embeddings')
        fig.tight_layout()
        fig.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Implementation of a Restricted Boltzmann Machine")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--epochs', action='store', type=int, default=1, help='Number of epochs of training')
    parser.add_argument('--lr', action='store', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--k', action='store', type=int, default=1, help='Learning rate')
    parser.add_argument('--bs', action='store', type=int, default=1, help='Batch size')
    parser.add_argument('--save', '-s', action='store_true', help="Save or not the RBM's weights")
    parser.add_argument('--save_path', action='store', type=str, help="Path to save the RBM's weights")
    parser.add_argument('--fit_cl', action='store_true', help="Train or not the classifier")
    parser.add_argument('--load_w', '-ldw', action='store_true', help="Load or not the RBM's weights from file")
    parser.add_argument('--w_path', action='store', type=str, help="Path to the RBM's weights")
    args = parser.parse_args()

    imgs, _, _, _ = load_mnist(path="MNIST/")
    rbm = RBM(n_visible=len(imgs[0]), mnist_path='MNIST/')
    if args.train:
        rbm.fit(epochs=args.epochs,
                lr=args.lr,
                k=args.k,
                bs=args.bs,
                save=args.save,
                save_path=args.save_path,
                fit_cl=args.fit_cl,
                save_cl=False,
                save_cl_path=None,
                show_feats=False)
    else:
        # load weights from file
        rbm.load_weights(args.w_path)
        # train and test classifier
        rbm.fit_classifier(load_boltz_weights=args.load_w, w_path=args.w_path)
        rbm.test_classifier()
        # plot confusion matrix
        rbm.confusion_matrix()
        # plot a reconstruction for each digit
        # indexes = []
        # curr = 0
        # while curr < 10:
        #     for i, label in enumerate(rbm.tr_labels):
        #         if label == curr:
        #             indexes.append(i)
        #             curr += 1
        #             break
        # fig, ax = plt.subplots(2, 10, figsize=(5, 2))
        # for i in range(20):
        #     if i < 10:
        #         ax[0, i].imshow(np.reshape(rbm.tr_imgs[indexes[i]], newshape=(28, 28)))
        #         ax[0, i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        #     else:
        #         v_sample = np.random.binomial(n=1, p=rbm.tr_imgs[indexes[i - 10]], size=len(rbm.tr_imgs[indexes[i - 10]]))
        #         _, h_sample = rbm.ph_v(v_sample)
        #         v_probs, _ = rbm.pv_h(h_sample)
        #         ax[1, i - 10].imshow(np.reshape(v_probs, newshape=(28, 28)))
        #         ax[1, i - 10].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        # fig.suptitle("Original images and their reconstructions")
        # fig.show()
        # show reconstructions of specific images
        # rbm.show_reconstruction(imgs[0])
        # rbm.show_reconstruction(imgs[1])
        # rbm.show_reconstruction(imgs[2])
        # rbm.show_reconstruction(imgs[3])
        # rbm.show_reconstruction(imgs[4])
