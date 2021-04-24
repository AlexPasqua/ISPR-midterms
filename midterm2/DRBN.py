import datetime
import math
import pickle

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utilities import load_mnist, sigmoid


# noinspection PyTypeChecker
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

    def forward(self, v_probs):
        v_sample = np.random.binomial(n=1, p=v_probs, size=len(v_probs))
        probs = [v_probs] + [None] * (self.nl - 1)
        samples = [v_sample] + [None] * (self.nl - 1)
        for i in range(1, self.nl):
            probs[i], samples[i] = self.forward_step(layer_idx=i, prev_sample=samples[i - 1])
        return probs, samples

    def backward(self, top_prob):
        top_sample = np.random.binomial(n=1, p=top_prob, size=len(top_prob))
        probs = [None] * (self.nl - 1) + [top_prob]
        samples = [None] * (self.nl - 1) + [top_sample]
        for i in reversed(range(self.nl - 1)):
            probs[i], samples[i] = self.backward_step(layer_idx=i, prev_sample=samples[i + 1])
        return probs, samples

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

    def fit(self, epochs, lr, k, bs=1, save=False):
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
