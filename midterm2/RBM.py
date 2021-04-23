import copy
import pickle
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
        # load mnist dataset
        self.tr_imgs, self.tr_labels, self.ts_imgs, self.ts_labels = load_mnist(mnist_path)
        # set weights and biases
        assert n_visible >= 0 and n_hidden >= 0
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        weights_shape = (n_hidden, n_visible)
        self.W = W if W is not None else np.random.uniform(-1, 1, size=weights_shape)
        self.bias_visible = bias_visible if bias_visible is not None else np.zeros(n_visible)
        self.bias_hidden = bias_hidden if bias_hidden is not None else np.zeros(n_hidden)
        assert self.W.shape == weights_shape and \
               n_visible == len(self.bias_visible) and \
               n_hidden == len(self.bias_hidden)
        # create classifier
        self.classifier = tf.keras.models.Sequential([
            Dense(units=50, activation='relu', input_dim=n_hidden),
            Dense(units=10, activation='softmax')
        ])

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
        # in case it doesn't enter the cycle, but it shouldn't happen
        v_prob, v_sample, h_prob = None, None, None
        for i in range(k):
            v_prob, v_sample = self.pv_h(h_sample)
            h_prob, h_sample = self.ph_v(v_sample)
        return v_prob, v_sample, h_prob, h_sample

    def contrastive_divergence(self, v_probs, k, lr):
        """
        Perform one step of contrastive divergence
        :param v_probs: vector of the visible units activations (non-binarized)
        :param k: (int) order of the Gibbs sampling
        :param lr: (float) learning rate (0 < lr <= 1)
        """
        # compute wake part
        v_sample = np.random.binomial(n=1, p=v_probs, size=len(v_probs))
        h_probs, h_sample = self.ph_v(v_sample)
        wake = np.outer(h_probs, v_probs)
        # compute dream part
        v_probs_gibbs, v_sample_gibbs, h_probs_gibbs, h_sample_gibbs = self.gibbs_sampling(h_sample, k)
        dream = np.outer(h_probs_gibbs, v_probs_gibbs)
        # weights update
        self.W += lr * np.subtract(wake, dream)
        self.bias_visible += lr * np.subtract(v_sample, v_sample_gibbs)
        self.bias_hidden += lr * np.subtract(h_sample, h_sample_gibbs)

    def fit(self, epochs, lr, k, batch_size=50, save=False, save_path=None, fit_cl=False, save_cl=False, save_cl_path=None, show_feats=False):
        """
        Perform model fitting by contrastive divergence
        :param epochs: (int) number of epochs of training
        :param lr: (float) learning rate (0 < lr <= 1)
        :param k: (int) order of the Gibbs sampling
        :param batch_size: (int) size of a batch/minibatch of training data
        :param save: (bool) if True, save the model's weights and biases
        :param save_path: (str) path where to save the model
        :param fit_cl: (bool) if True, fit the classifier on the encodings
        :param save_cl: (book) if True, save the classifier's weights
        :param save_cl_path: (str) path where to save the classifier
        :param show_feats: (bool) if True, plot the learnt features (reshaped rows of weights matrix)
        """
        assert epochs > 0 and 0 < lr <= 1 and k > 0

        # uncomment to shorten the training set for debugging purposes
        # self.tr_imgs, self.tr_labels = self.tr_imgs[:100], self.tr_labels[:100]

        # iterate over epochs
        indexes = list(range(len(self.tr_imgs)))
        for ep in range(epochs):
            # shuffle the data
            np.random.shuffle(indexes)
            self.tr_imgs = self.tr_imgs[indexes]
            self.tr_labels = self.tr_labels[indexes]
            # iterate over data samples
            for i in tqdm(range(len(self.tr_labels))):
                self.contrastive_divergence(v_probs=self.tr_imgs[i], k=k, lr=lr)

        if save:
            self.save_model(datetime.now().strftime("rbm_%d-%m-%y_%H-%M") if save_path is None else save_path)
        if show_feats:
            self.show_learnt_features()
        if fit_cl:
            # train the classifier on the embeddings
            self.fit_classifier(save=save_cl, save_path=save_cl_path)

    def fit_classifier(self, load_rbm_weights=False, w_path=None, save=True, save_path=None):
        """
        Train the classifier on the embeddings of the RBM
        :param load_rbm_weights: (bool) if True, load the weights of the RBM from a file
        :param w_path: (str) path where the RBM's weights are stored
        :param save: (bool) if True, save the classifier's weights
        :param save_path: (str) path where the classifier's weights are stored
        :returns hist: training history
        """
        # load weights of the RBM from file
        if load_rbm_weights:
            self.load_weights(w_path)
        # create a training set by encoding all the training images
        tr_set = []
        for i in range(len(self.tr_labels)):
            encoding = sigmoid(np.add(np.matmul(self.W, self.tr_imgs[i]), self.bias_hidden))
            encoding = np.random.binomial(n=1, p=encoding, size=len(encoding))
            tr_set.append(encoding)
        # 1-hot encoding of the labels
        train_labels = tf.stack(to_categorical(self.tr_labels, 10))
        tr_set = tf.stack(tr_set)
        # compile and fit the classifier
        self.classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics='categorical_accuracy')
        hist = self.classifier.fit(x=tr_set, y=train_labels, epochs=5)
        # save the classifier's weights
        if save:
            save_path = datetime.now().strftime("classifier_%d-%m-%y_%H-%M.h5") if save_path is None else save_path
            self.classifier.save_weights(save_path)
        return hist

    def test_classifier(self, test_images=None, test_labels=None):
        """
        Evaluate the classifier
        :param test_images: specific set of test images (optional)
        :param test_labels: specific set of test labels (optional)
        :return: results of the test -> loss and accuracy
        """
        # if a specific set of test images and labels is NOT specified, use the MNIST test set
        if test_images is None:
            test_images = self.ts_imgs
            test_labels = copy.deepcopy(self.ts_labels)
        # create a test set by encoding all the test images
        test_encodings = []
        for i in range(len(test_labels)):
            encoding = sigmoid(np.add(np.matmul(self.W, test_images[i]), self.bias_hidden))
            encoding = np.random.binomial(n=1, p=encoding, size=len(encoding))
            test_encodings.append(encoding)
        # 1-hot encoding of the test labels
        test_encodings = tf.stack(test_encodings)
        test_labels = tf.stack(to_categorical(test_labels))
        # classifier evaluation
        res = self.classifier.evaluate(x=test_encodings, y=test_labels, return_dict=True)
        return res

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
            assert len(weights[0]) == len(bias_visible) == self.n_visible and \
                   len(weights[:, 0]) == len(bias_hidden) == self.n_hidden
            self.W = weights
            self.bias_visible = bias_visible
            self.bias_hidden = bias_hidden
