import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from datetime import datetime
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
        :raises AssertionError: if there are incompatibilities between parameters
        """
        assert n_visible >= 0 and n_hidden >= 0
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        weights_shape = (n_hidden, n_visible)
        self.W = W if W is not None else np.random.uniform(-1, 1, size=weights_shape)
        self.bias_visible = bias_visible if bias_visible is not None else np.zeros(n_visible)
        self.bias_hidden = bias_hidden if bias_hidden is not None else np.zeros(n_hidden)
        assert self.W.shape == weights_shape and n_visible == len(self.bias_visible) and n_hidden == len(
            self.bias_hidden)
        self.classifier = MLPClassifier(
            hidden_layer_sizes=(10, 10),
            activation='relu',
            solver='adam',
            learning_rate_init=0.05,
            max_iter=200,
            verbose=True
        )

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
        v_sample = None  # in case it doesn't enter the cycle, but it shouldn't happen
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

    def fit(self, epochs, lr, k, mnist_path=None, save=True, save_path=None, fit_classifier=False):
        """ Perform model fitting """
        assert epochs >= 0 and lr > 0 and k > 0
        train_images, train_labels, test_images, test_labels = load_mnist(mnist_path)

        # TODO: remove following line -> it's for shortening the training set
        # train_images, train_labels = train_images[:100], train_labels[:100]

        for ep in range(epochs):
            for i in tqdm(range(len(train_labels))):
                self.contrastive_divergence(v_prob=train_images[i], k=k, lr=lr)

        if save:
            self.save_model(datetime.now().strftime("rbm_%d-%m-%y_%H-%M") if save_path is None else save_path)

        if fit_classifier:
            # TODO: fix body of if statement -> messy for now
            self.fit_classifier(mnist_path=mnist_path)
            self.test_classifier(mnist_path=mnist_path)
            # ts_img = sigmoid(np.add(np.matmul(self.W, test_images[0]), self.bias_hidden))
            # prediction = np.argmax(self.classifier.predict_proba(ts_img.reshape(1, -1)))
            # print("Predicted: ", prediction)
            # img = np.reshape(test_images[0], newshape=(28, 28))
            # plt.imshow(img)
            # plt.show()

    def fit_classifier(self, load_rbm_weights=False, w_path=None, mnist_path=None, save=True, save_path=None):
        if load_rbm_weights:
            self.load_weights(w_path)
        train_images, train_labels, _, _ = load_mnist(mnist_path)
        tr_set = []
        for i in range(len(train_labels)):
            encoding = sigmoid(np.add(np.matmul(self.W, train_images[i]), self.bias_hidden))
            tr_set.append(encoding)
        self.classifier.fit(tr_set, train_labels)
        if save:
            save_path = datetime.now().strftime("classifier_%d-%m-%y_%H-%M") if save_path is None else save_path
            with open(save_path, 'wb') as f:
                pickle.dump(self.classifier, f)

    def test_classifier(self, test_images=None, test_labels=None, mnist_path=None):
        if test_images is None:
            _, _, test_images, test_labels = load_mnist(mnist_path)
        test_encodings = []
        for i in range(len(test_labels)):
            enc = sigmoid(np.add(np.matmul(self.W, test_images[0]), self.bias_hidden))
            test_encodings.append(enc)

    def show_learnt_features(self):
        # TODO: improve and finish this method
        # plot the weights to see learnt features
        # not sure if it's right
        for i in range(3):
            fig, ax = plt.subplots(1, 2)
            img = np.reshape(self.W[i], newshape=(28, 28))
            ax[0].imshow(img)
            img = np.reshape(self.W[:][i], newshape=(28, 28))
            ax[1].imshow(img)
            fig.show()

    def save_model(self, path):
        """
        Saves the model on a pickle file
        :param path: the file where to save the model
        """
        path = path if (path.endswith('.pickle') or path.endswith('.pkl')) else path + '.pickle'
        dump_dict = {'n_visible': self.n_visible,
                     'n_hidden': self.n_hidden,
                     'weights': self.W,
                     'bias_visible': self.bias_visible,
                     'bias_hidden': self.bias_hidden}
        with open(path, 'wb') as f:
            pickle.dump(dump_dict, f)

    def load_weights(self, path):
        """
        Loads the model's weights (and biases) from a pickle file.
        The model's architecture must be compatible with the weights being loaded.
        :param path: the path to the json file where the weights are stored
        :raises FileNotFoundError: if path does not correspond to an existing file
        :raises AssertionError: if the shape of the weights and biases is not compatible with model's architecture
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            weights = data['weights']
            bias_visible = data['bias_visible']
            bias_hidden = data['bias_hidden']
            assert len(weights[0]) == len(bias_visible) == self.n_visible \
                   and len(weights[:, 0]) == len(bias_hidden) == self.n_hidden
            self.W = weights
            self.bias_visible = bias_visible
            self.bias_hidden = bias_hidden
