import copy
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.utils.np_utils import to_categorical
from tqdm import tqdm
from datetime import datetime
from utilities import load_mnist, sigmoid


class DRBN:
    """
    Implementation of a Deep Restricted Boltzmann Network
    reference paper: https://arxiv.org/pdf/1611.07917.pdf
    """

    def __init__(self, hl_sizes, v_size, mnist_path=None):
        """
        Constructor
        :param hl_sizes: (iterable<int>) sizes of the layers (input layer excluded)
        :param v_size: (int) size of the visible (input) layer
        :param mnist_path: (str) path to the MNIST dataset
        """
        # check parameters
        assert [size > 0 for size in hl_sizes] and v_size > 0
        n_hl = len(hl_sizes)    # number of hidden layers
        self.nl = n_hl + 1      # number of layers
        self.units_per_layer = np.concatenate(([v_size], hl_sizes))
        # load mnist dataset
        self.tr_imgs, self.tr_labels, self.ts_imgs, self.ts_labels = load_mnist(mnist_path)
        # initialize weights and biases
        sizes = np.concatenate(([v_size], hl_sizes))
        self.W_matrs = [np.random.uniform(-1, 1, size=(sizes[i+1], sizes[i])) for i in range(n_hl)]
        self.biases = [np.zeros(sizes[i]) for i in range(n_hl + 1)]
        # create classifier
        self.classifier = tf.keras.models.Sequential([
            Dense(units=50, activation='relu', input_dim=sizes[-1]),
            Dense(units=10, activation='softmax')
        ])

    def forward_step(self, layer_idx, prev_sample):
        """
        Perform one forward step
        :param layer_idx: index of the layer we are going to
        :param prev_sample: previous layer's binarized activations
        :return: the next layer's probabilities and samples (i.e. binarized activations)
        """
        probs = sigmoid(np.add(np.matmul(self.W_matrs[layer_idx - 1], prev_sample), self.biases[layer_idx]))
        samples = np.random.binomial(n=1, p=probs, size=len(probs))
        return probs, samples

    def backward_step(self, layer_idx, prev_sample):
        """
        Perform one backward step
        :param layer_idx: index of the layer we are going to
        :param prev_sample: previous layer's binarized activations
        :return: the probabilities and samples (i.e. binarized activations) of the layer we're going to
        """
        probs = sigmoid(np.add(np.matmul(prev_sample, self.W_matrs[layer_idx]), self.biases[layer_idx]))
        samples = np.random.binomial(n=1, p=probs, size=len(probs))
        return probs, samples

    # noinspection PyTypeChecker
    def forward(self, v_probs):
        v_sample = np.random.binomial(n=1, p=v_probs, size=len(v_probs))
        probs = [v_probs] + [None] * (self.nl - 1)
        samples = [v_sample] + [None] * (self.nl - 1)
        for i in range(1, self.nl):
            probs[i], samples[i] = self.forward_step(layer_idx=i, prev_sample=samples[i - 1])
        return probs, samples

    # noinspection PyTypeChecker
    def backward(self, top_prob):
        top_sample = np.random.binomial(n=1, p=top_prob, size=len(top_prob))
        probs = [None] * (self.nl - 1) + [top_prob]
        samples = [None] * (self.nl - 1) + [top_sample]
        for i in reversed(range(self.nl - 1)):
            probs[i], samples[i] = self.backward_step(layer_idx=i, prev_sample=samples[i + 1])
        return probs, samples

    # noinspection PyTypeChecker
    def gibbs_sampling(self, top_prob, k):
        samples = None  # in case it doesn't enter the cycle, but it shouldn't happen
        probs = [None] * (self.nl - 1) + [top_prob]
        for i in range(k):
            probs, samples = self.backward(probs[-1])
            probs, samples = self.forward(probs[0])
        return probs, samples

    def persistent_contrastive_divergence(self, v_probs, k):
        # compute wake terms
        probs, samples = self.forward(v_probs)
        wake_terms = [np.outer(probs[i + 1], probs[i]) for i in range(self.nl - 1)]
        # compute dream terms
        probs_gibbs, samples_gibbs = self.gibbs_sampling(top_prob=probs[-1], k=k)
        dream_terms = [np.outer(probs_gibbs[i + 1], probs_gibbs[i]) for i in range(self.nl - 1)]
        # compute deltas
        delta_W = [np.subtract(wake_terms[i], dream_terms[i]) for i in range(self.nl - 1)]
        delta_b = [np.subtract(samples[i], samples_gibbs[i]) for i in range(self.nl)]
        return delta_W, delta_b

    def fit(self, epochs, lr, k, bs=1, save=False, save_path=None, fit_cl=False, save_cl=False, save_cl_path=None):
        assert epochs > 0 and 0 < lr <= 1 and k > 0
        n_imgs = len(self.tr_labels)
        bs = n_imgs if bs == 'batch' or bs > n_imgs else bs
        disable_tqdm = (False, True) if bs < n_imgs else (True, False)  # for progress bars
        indexes = list(range(len(self.tr_imgs)))
        # iterate over epochs
        for ep in range(epochs):
            # shuffle the data
            np.random.shuffle(indexes)
            self.tr_imgs = self.tr_imgs[indexes]
            self.tr_labels = self.tr_labels[indexes]

            # cycle through batches
            for batch_idx in tqdm(range(math.ceil(len(self.tr_labels) / bs)), disable=disable_tqdm[0]):
                delta_W = [np.zeros(shape=self.W_matrs[i].shape) for i in range(self.nl - 1)]
                delta_b = [np.zeros(shape=(len(self.biases[i]),)) for i in range(self.nl)]
                start = batch_idx * bs
                end = start + bs
                batch_imgs = self.tr_imgs[start: end]

                # cycle through patterns within a batch
                for img in tqdm(batch_imgs, disable=disable_tqdm[1]):
                    dW, db = self.persistent_contrastive_divergence(v_probs=img, k=k)
                    for i in range(self.nl - 1):
                        delta_W[i] = np.add(delta_W[i], dW[i])
                        delta_b[i] = np.add(delta_b[i], db[i])
                    delta_b[-1] = np.add(delta_b[-1], db[-1])   # update last layer's bias

                # weights update
                rescaled_lr = lr / bs
                for i in range(self.nl - 1):
                    self.W_matrs[i] = np.add(self.W_matrs[i], np.multiply(rescaled_lr, delta_W[i]))
                    self.biases[i] = np.add(self.biases[i], np.multiply(rescaled_lr, delta_b[i]))
                self.biases[-1] = np.add(self.biases[-1], np.multiply(rescaled_lr, delta_b[-1]))    # update last layer's bias
        if save:
            self.save_model(datetime.now().strftime("rbm_%d-%m-%y_%H-%M") if save_path is None else save_path)
        if fit_cl:
            self.fit_classifier()

    def fit_classifier(self, load_drbn_weights=False, w_path=None, save=False, save_path=None):
        """
        Train the classifier on the embeddings of the RBM
        :param load_drbn_weights: (bool) if True, load the weights of the DRBN from a file
        :param w_path: (str) path where the DRBN's weights are stored
        :param save: (bool) if True, save the classifier's weights
        :param save_path: (str) path where the classifier's weights are stored
        :returns hist: training history
        """
        # load weights of the RBM from file
        if load_drbn_weights:
            self.load_weights(w_path)
        # create a training set by encoding all the training images
        tr_set = []
        for i in range(len(self.tr_labels)):
            _, encoding = self.forward(v_probs=self.tr_imgs[i])
            tr_set.append(encoding[-1])
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
        if test_images is not None and test_labels is not None:
            assert len(test_images) == len(test_labels)
        elif not(test_images is None and test_labels is None):
            raise RuntimeWarning("The number of test images differs from the number of test labels. MNIST test set is going to be used")
        if test_images is None:
            test_images = self.ts_imgs
            test_labels = copy.deepcopy(self.ts_labels)
        # create a test set by encoding all the test images
        test_encodings = []
        for i in range(len(test_labels)):
            _, encoding = self.forward(v_probs=test_images[i])
            test_encodings.append(encoding[-1])
        # 1-hot encoding of the test labels
        test_encodings = tf.stack(test_encodings)
        test_labels = tf.stack(to_categorical(test_labels))
        # classifier evaluation
        res = self.classifier.evaluate(x=test_encodings, y=test_labels, return_dict=True)
        return res

    def show_reconstruction(self, img):
        """
        Plots a reconstruction of an image
        :param img: (vector) the image to reconstruct
        """
        probs, samples = self.forward(v_probs=img)
        probs, samples = self.backward(top_prob=probs[-1])
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(np.reshape(img, newshape=(28, 28)))
        ax[0].set_title('Original image')
        ax[1].imshow(np.reshape(probs[0], newshape=(28, 28)))
        ax[1].set_title('Reconstructed image')
        fig.suptitle('Reconstruction')
        fig.tight_layout()
        fig.show()

    def save_model(self, path):
        """
        Saves the model on a pickle file
        :param path: the file where to save the model
        """
        path = path if (path.endswith('.pickle') or path.endswith('.pkl')) else path + '.pickle'
        dump_dict = {'n_layers': self.nl,
                     'units_per_layer': self.units_per_layer,
                     'weights_matrices': self.W_matrs,
                     'biases': self.biases}
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
            weights_matrices = data['weights_matrices']
            biases = data['biases']
            assert len(biases) == len(weights_matrices) + 1 == self.nl
            for i in range(self.nl - 1):
                weights_shape = np.shape(weights_matrices[i])
                assert weights_shape[0] == self.units_per_layer[i + 1] and \
                    weights_shape[1] == len(biases[i]) == self.units_per_layer[i]
                self.W_matrs[i] = weights_matrices[i]
                self.biases[i] = biases[i]
            self.biases[-1] = biases[-1]
