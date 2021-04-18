import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utilities import load_mnist, sigmoid


class RBM:
    """ Implementation of a Restricted Boltzmann Machine """

    def __init__(self, n_visible, n_hidden=100, W=None, bias_visible=None, bias_hidden=None):
        """
        Constructor
        :param n_visible: number of visible units
        :param n_hidden: number of hidden units
        :param W: weights matrix
        :param bias_visible: bias vector for visible layer
        :param bias_hidden: bias vector for hidden layer
        """
        assert n_visible >= 0 and n_hidden >= 0
        weights_shape = (n_hidden, n_visible)
        self.W = W if W is not None else np.random.uniform(-1, 1, size=weights_shape)
        self.bias_visible = bias_visible if bias_visible is not None else np.zeros(n_visible)
        self.bias_hidden = bias_hidden if bias_hidden is not None else np.zeros(n_hidden)
        assert self.W.shape == weights_shape and n_visible == len(self.bias_visible) and n_hidden == len(
            self.bias_hidden)

    def ph_v(self, v_sample):
        """ Compute conditional probability of the hidden units given the visible ones """
        h_probs = sigmoid(np.add(np.matmul(self.W, v_sample), self.bias_hidden))
        h_samples = np.random.binomial(n=1, p=h_probs, size=len(h_probs))
        return h_samples

    def pv_h(self, h_sample):
        """ Compute conditional probability of the visible units given the hidden ones """
        v_probs = sigmoid(np.add(np.matmul(h_sample, self.W), self.bias_visible))
        v_samples = np.random.binomial(n=1, p=v_probs, size=len(v_probs))
        return v_samples

    def gibbs_sampling(self, h_sample, k):
        """ Performs Gibbs sampling """
        v_sample = None     # in case it doesn't enter the cycle, but it shouldn't happen
        for i in range(k):
            v_prob = sigmoid(np.add(np.matmul(h_sample, self.W), self.bias_visible))
            v_sample = np.random.binomial(n=1, p=v_prob, size=len(v_prob))
            h_prob = sigmoid(np.add(np.matmul(self.W, v_sample), self.bias_hidden))
            h_sample = np.random.binomial(n=1, p=h_prob, size=len(h_prob))
        return v_sample, h_sample

    def contrastive_divergence(self, v_prob, k, lr):
        """ Perform one step of contrastive divergence """
        v_sample = np.random.binomial(n=1, p=v_prob, size=len(v_prob))
        h_sample = self.ph_v(v_sample)
        wake = np.outer(h_sample, v_sample)
        v_sample_gibbs, h_sample_gibbs = self.gibbs_sampling(h_sample, k)
        dream = np.outer(h_sample_gibbs, v_sample_gibbs)
        # weights update
        self.W += lr * np.subtract(wake, dream)
        self.bias_visible += lr * np.subtract(v_sample, v_sample_gibbs)
        self.bias_hidden += lr * np.subtract(h_sample, h_sample_gibbs)

    def fit(self, epochs, lr, k, mnist_path=None):
        """ Perform model fitting """
        assert epochs >= 0 and lr > 0 and k > 0
        train_images, train_labels, test_images, test_labels = load_mnist(mnist_path)

        # TODO: remove following line -> it's for shortening the training set
        # train_images, train_labels = train_images[:100], train_labels[:100]

        for ep in range(epochs):
            for i in tqdm(range(len(train_labels))):
                self.contrastive_divergence(v_prob=train_images[i], k=k, lr=lr)

        # plot the weights to see learnt features
        # not sure if it's right
        for i in range(10):
            fig, ax = plt.subplots(1, 2)
            img = np.reshape(self.W[i], newshape=(28, 28))
            ax[0].imshow(img)
            img = np.reshape(self.W[:][i], newshape=(28, 28))
            ax[1].imshow(img)
            fig.show()
