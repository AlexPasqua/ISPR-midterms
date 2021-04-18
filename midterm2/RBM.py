import numpy as np
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
        :param k: param k for Gibb's sampling
        """
        assert n_visible >= 0 and n_hidden >= 0
        # self.n_visible = n_visible
        # self.n_hidden = n_hidden
        weights_shape = (n_hidden, n_visible)
        self.W = W if W is not None else np.random.uniform(-1, 1, size=weights_shape)
        self.bias_visible = bias_visible if bias_visible is not None else np.zeros(n_visible)
        self.bias_hidden = bias_hidden if bias_hidden is not None else np.zeros(n_hidden)
        assert self.W.shape == weights_shape and n_visible == len(self.bias_visible) and n_hidden == len(self.bias_hidden)

    def gibbs_sampling(self, v, k):
        for i in range(k):
            h_probs = sigmoid(np.add(np.matmul(self.W, v), self.bias_hidden))
            print(h_probs.shape)
            # h_samples = np.random.binomial(n=1, p=h_probs, size=len(h_probs))
            # reconstruct

    def contrastive_divergence(self, v, k):
        self.gibbs_sampling(v, k)

    def fit(self, epochs, lr, k):
        assert epochs >= 0 and lr > 0 and k > 0
        train_images, train_labels, test_images, test_labels = load_mnist()
        for ep in range(epochs):
            for i in range(len(train_labels)):
                v = train_images[i]
                self.contrastive_divergence(v, k)
