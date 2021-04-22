import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.utils.np_utils import to_categorical
from datetime import datetime
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
        self.classifier = tf.keras.models.Sequential([
            Dense(units=50, activation='relu', input_dim=n_hidden),
            Dense(units=10, activation='softmax')
        ])

    def ph_v(self, v_sample):
        """ Compute conditional probability of the hidden units given the visible ones """
        h_probs = sigmoid(np.add(np.matmul(self.W, v_sample), self.bias_hidden))
        h_samples = np.random.binomial(n=1, p=h_probs, size=len(h_probs))
        return h_probs, h_samples

    def pv_h(self, h_sample):
        """ Compute conditional probability of the visible units given the hidden ones """
        v_probs = sigmoid(np.add(np.matmul(h_sample, self.W), self.bias_visible))
        v_samples = np.random.binomial(n=1, p=v_probs, size=len(v_probs))
        return v_probs, v_samples

    def gibbs_sampling(self, h_sample, k):
        """ Performs Gibbs sampling """
        # in case it doesn't enter the cycle, but it shouldn't happen
        v_prob, v_sample, h_prob = None, None, None
        for i in range(k):
            v_prob = sigmoid(np.add(np.matmul(h_sample, self.W), self.bias_visible))
            v_sample = np.random.binomial(n=1, p=v_prob, size=len(v_prob))
            h_prob = sigmoid(np.add(np.matmul(self.W, v_sample), self.bias_hidden))
            h_sample = np.random.binomial(n=1, p=h_prob, size=len(h_prob))
        return v_prob, v_sample, h_prob, h_sample

    def contrastive_divergence(self, v_prob, k, lr):
        """ Perform one step of contrastive divergence """
        v_sample = np.random.binomial(n=1, p=v_prob, size=len(v_prob))
        h_probs, h_sample = self.ph_v(v_sample)
        wake = np.outer(h_probs, v_sample)
        v_probs_gibbs, v_sample_gibbs, h_probs_gibbs, h_sample_gibbs = self.gibbs_sampling(h_sample, k)
        dream = np.outer(h_probs_gibbs, v_sample_gibbs)
        # weights update
        self.W += lr * np.subtract(wake, dream)
        self.bias_visible += lr * np.subtract(v_sample, v_sample_gibbs)
        self.bias_hidden += lr * np.subtract(h_sample, h_sample_gibbs)

    def fit(self, epochs, lr, k, mnist_path=None, save=True, save_path=None, fit_classifier=False, show_feats=False):
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
        if show_feats:
            self.show_learnt_features()

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
            encoding = np.random.binomial(n=1, p=encoding, size=len(encoding))
            tr_set.append(encoding)

        train_labels = tf.stack(to_categorical(train_labels, 10))
        tr_set = tf.stack(tr_set)
        self.classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
        self.classifier.fit(
            x=tr_set,
            y=train_labels,
            epochs=5,
            use_multiprocessing=True,
            workers=4
        )
        if save:
            save_path = datetime.now().strftime("classifier_%d-%m-%y_%H-%M") if save_path is None else save_path
            self.classifier.save(save_path)

    def test_classifier(self, test_images=None, test_labels=None, mnist_path=None):
        if test_images is None:
            _, _, test_images, test_labels = load_mnist(mnist_path)
        test_encodings = []
        for i in range(len(test_labels)):
            encoding = sigmoid(np.add(np.matmul(self.W, test_images[0]), self.bias_hidden))
            encoding = np.random.binomial(n=1, p=encoding, size=len(encoding))
            test_encodings.append(encoding)
        test_encodings = tf.stack(test_encodings)
        test_labels = tf.stack(to_categorical(test_labels))
        res = self.classifier.evaluate(
            x=test_encodings,
            y=test_labels,
            return_dict=True
        )
        for k, v in res.items():
            print(f"{k}: {v}")

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

    def show_embedding(self, img=None):
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
