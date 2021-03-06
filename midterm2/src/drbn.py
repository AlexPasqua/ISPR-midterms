import argparse
import copy
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns
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
        n_hl = len(hl_sizes)  # number of hidden layers
        self.nl = n_hl + 1  # number of layers
        self.units_per_layer = np.concatenate(([v_size], hl_sizes))
        # load mnist dataset
        self.tr_imgs, self.tr_labels, self.ts_imgs, self.ts_labels = load_mnist(mnist_path)
        # initialize weights and biases
        sizes = np.concatenate(([v_size], hl_sizes))
        self.W_matrs = [np.random.uniform(-1, 1, size=(sizes[i + 1], sizes[i])) for i in range(n_hl)]
        self.biases = [np.zeros(sizes[i]) for i in range(n_hl + 1)]
        self.particle = np.zeros(sizes[0])
        # create classifier
        self.classifier = tf.keras.models.Sequential([
            Dense(units=200, activation='relu', input_dim=sizes[-1]),
            Dense(units=100, activation='relu'),
            Dense(units=50, activation='relu'),
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

    def forward(self, v_probs):
        """ Compute conditional probability and samples of the hidden layer given the visible one, for each RBM """
        v_sample = np.random.binomial(n=1, p=v_probs, size=len(v_probs))
        probs = [v_probs] + [None] * (self.nl - 1)
        samples = [v_sample] + [None] * (self.nl - 1)
        for i in range(1, self.nl):
            probs[i], samples[i] = self.forward_step(layer_idx=i, prev_sample=samples[i - 1])
        return probs, samples

    def backward(self, top_prob):
        """ Compute conditional probability and samples of the visible layers given the hidden one, for each RBM """
        top_sample = np.random.binomial(n=1, p=top_prob, size=len(top_prob))
        probs = [None] * (self.nl - 1) + [top_prob]
        samples = [None] * (self.nl - 1) + [top_sample]
        for i in reversed(range(self.nl - 1)):
            probs[i], samples[i] = self.backward_step(layer_idx=i, prev_sample=samples[i + 1])
        return probs, samples

    def gibbs_sampling(self, top_prob, k):
        """
        Performs Gibbs sampling
        :param top_prob: binary vector of the activations of the last RBM's hidden layer
        :param k: order of the Gibbs sampling
        """
        samples = None  # in case it doesn't enter the cycle, but it shouldn't happen
        probs = [None] * (self.nl - 1) + [top_prob]
        for i in range(k):
            probs, samples = self.backward(probs[-1])
            probs, samples = self.forward(probs[0])
        return probs, samples

    def compute_deltas(self, wake_terms, dream_terms, samples, samples_gibbs):
        """ Compute delta weights and delta biases """
        delta_W = [np.subtract(wake_terms[i], dream_terms[i]) for i in range(self.nl - 1)]
        delta_b = [np.subtract(samples[i], samples_gibbs[i]) for i in range(self.nl)]
        return delta_W, delta_b

    def persistent_contrastive_divergence(self, v_probs, k=1, persist=False):
        """
        Perform one step of persistent contrastive divergence, with 1 Gibbs chain
        :param v_probs: vector of the visible units activations (non-binarized)
        :param k: (int) order of the Gibbs sampling
        :param persist: (bool) if False, reset the Gibbs sampling chain
        """
        # compute wake terms
        probs, samples = self.forward(v_probs)
        wake_terms = [np.outer(probs[i + 1], probs[i]) for i in range(self.nl - 1)]
        # NOTE: the 1st time this method is called for each batch, it will enter the 'else'
        # so self.particle will be correctly initialized for the next iterations (enter 'if')
        if persist:
            top_prob = self.forward(self.particle)[0][-1]
            probs_gibbs, samples_gibbs = self.gibbs_sampling(top_prob=top_prob, k=k)
            self.particle = probs_gibbs[0]
        else:
            probs_gibbs, samples_gibbs = self.gibbs_sampling(top_prob=probs[-1], k=k)
            self.particle = probs_gibbs[0]
        # compute dream terms
        dream_terms = [np.outer(probs_gibbs[i + 1], probs_gibbs[i]) for i in range(self.nl - 1)]
        # compute deltas and return
        return self.compute_deltas(wake_terms, dream_terms, samples, samples_gibbs)

    def contrastive_divergence(self, v_probs, k):
        """
        Perform one step of contrastive divergence
        :param v_probs: vector of the visible units activations (non-binarized)
        :param k: (int) order of the Gibbs sampling
        """
        # compute wake terms
        probs, samples = self.forward(v_probs)
        wake_terms = [np.outer(probs[i + 1], probs[i]) for i in range(self.nl - 1)]
        # compute dream terms
        probs_gibbs, samples_gibbs = self.gibbs_sampling(top_prob=probs[-1], k=k)
        dream_terms = [np.outer(probs_gibbs[i + 1], probs_gibbs[i]) for i in range(self.nl - 1)]
        # compute deltas and return
        return self.compute_deltas(wake_terms, dream_terms, samples, samples_gibbs)

    def fit(self, alg='cd', epochs=1, lr=0.1, k=1, bs=1, save=False, save_path=None, fit_cl=False, save_cl=False,
            save_cl_path=None):
        """
        Perform model fitting by contrastive divergence
        :param alg: (str in {'cd', 'pcd'}) decide whether to use CD or PCD algorithm
        :param epochs: (int) number of epochs of training
        :param lr: (float) learning rate (0 < lr <= 1)
        :param k: (int) order of the Gibbs sampling
        :param bs: (int) size of a batch/minibatch of training data
        :param save: (bool) if True, save the model's weights and biases
        :param save_path: (str) path where to save the model
        :param fit_cl: (bool) if True, fit the classifier on the encodings
        :param save_cl: (book) if True, save the classifier's weights
        :param save_cl_path: (str) path where to save the classifier
        """
        assert epochs > 0 and 0 < lr <= 1 and k > 0 and alg in ('cd', 'pcd')
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
            persist = False
            # cycle through batches
            for batch_idx in tqdm(range(math.ceil(len(self.tr_labels) / bs)), disable=disable_tqdm[0]):
                delta_W = [np.zeros(shape=self.W_matrs[i].shape) for i in range(self.nl - 1)]
                delta_b = [np.zeros(shape=(len(self.biases[i]),)) for i in range(self.nl)]
                start = batch_idx * bs
                end = start + bs
                batch_imgs = self.tr_imgs[start: end]
                # cycle through patterns within a batch
                for i, img in tqdm(enumerate(batch_imgs), disable=disable_tqdm[1]):
                    if alg == 'cd':
                        dW, db = self.contrastive_divergence(v_probs=img, k=k)
                    else:
                        dW, db = self.persistent_contrastive_divergence(v_probs=img, k=k, persist=persist)
                        persist = True
                    for j in range(self.nl - 1):
                        delta_W[j] = np.add(delta_W[j], dW[j])
                        delta_b[j] = np.add(delta_b[j], db[j])
                    delta_b[-1] = np.add(delta_b[-1], db[-1])  # update last layer's bias
                # weights update
                rescaled_lr = lr / bs
                for i in range(self.nl - 1):
                    self.W_matrs[i] = np.add(self.W_matrs[i], np.multiply(rescaled_lr, delta_W[i]))
                    self.biases[i] = np.add(self.biases[i], np.multiply(rescaled_lr, delta_b[i]))
                self.biases[-1] = np.add(self.biases[-1],
                                         np.multiply(rescaled_lr, delta_b[-1]))  # update last layer's bias
        if save:
            self.save_model(datetime.now().strftime("drbn_%d-%m-%y_%H-%M") if save_path is None else save_path)
        if fit_cl:
            self.fit_classifier()

    def encode(self, images=None):
        """
        Creates the encodings for the images to feed the classifier
        :param images: (str or array-like) it specifies what images to encode:
            'train': encode MNIST training images
            'test': encode MNIST test images
            array-like: the images themselves as a bi-dimensional array (matrix) where each row is an image
        :return: the encoded images as a matrix (each row is an image)
        """
        images = self.tr_imgs if images == 'train' else (self.ts_imgs if images == 'test' else images)
        encodings = []
        for i in range(len(images)):
            _, enc = self.forward(v_probs=images[i])
            encodings.append(enc[-1])
        return encodings

    def fit_classifier(self, load_boltz_weights=False, w_path=None, save=False, save_path=None):
        """
        Train the classifier on the embeddings of the DRBN
        :param load_boltz_weights: (bool) if True, load the weights of the DRBN from a file
        :param w_path: (str) path where the DRBN's weights are stored
        :param save: (bool) if True, save the classifier's weights
        :param save_path: (str) path where the classifier's weights are stored
        :returns hist: training history
        """
        # load weights of the DRBN from file
        if load_boltz_weights:
            self.load_weights(w_path)
        # create a training set by encoding all the training images
        tr_set = self.encode('train')
        # 1-hot encoding of the labels
        train_labels = tf.stack(to_categorical(self.tr_labels, 10))
        tr_set = tf.stack(tr_set)
        # compile and fit the classifier
        self.classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
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
        elif not (test_images is None and test_labels is None):
            raise RuntimeWarning(
                "The number of test images differs from the number of test labels. MNIST test set is going to be used")
        if test_images is None:
            test_images = self.ts_imgs
            test_labels = copy.deepcopy(self.ts_labels)
        # create a test set by encoding all the test images
        test_encodings = self.encode('test')
        # 1-hot encoding of the test labels
        test_encodings = tf.stack(test_encodings)
        test_labels = tf.stack(to_categorical(test_labels))
        # classifier evaluation
        res = self.classifier.evaluate(x=test_encodings, y=test_labels, return_dict=True)
        return res

    def confusion_matrix(self, test_images=None, test_labels=None):
        """
        Plot confusion matrix
        :param test_images: list of specific images to test the classifier with
        :param test_labels: list of corresponding labels of test_images
        """
        # if a specific set of test images and labels is NOT specified, use the MNIST test set
        if test_images is not None and test_labels is not None:
            assert len(test_images) == len(test_labels)
        elif not (test_images is None and test_labels is None):
            raise RuntimeWarning(
                "The number of test images differs from the number of test labels. MNIST test set is going to be used")
        if test_images is None:
            test_images = self.ts_imgs
            test_labels = copy.deepcopy(self.ts_labels)
        # create a test set by encoding all the test images
        test_encodings = []
        for i in range(len(test_images)):
            _, encoding = self.forward(v_probs=test_images[i])
            test_encodings.append(encoding[-1])
        predictions = self.classifier.predict(tf.stack(test_encodings))
        predictions = np.argmax(predictions, axis=1)
        conf_matr = confusion_matrix(y_true=test_labels, y_pred=predictions)
        plt.figure(figsize=(7, 5))
        sns.heatmap(conf_matr, annot=True, annot_kws={'size': 8})
        plt.xlabel('True labels')
        plt.ylabel('Predictions')
        plt.title('Confusion matrix')
        plt.tight_layout()
        plt.show()

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Implementation of a Deep Restricted Boltzmann Network")
    parser.add_argument('--hl_sizes', action='store', type=int, nargs='+', default=(100,))
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--alg', action='store', type=str, default='cd', help='Type of algorithm to use {cd, pcd}')
    parser.add_argument('--epochs', action='store', type=int, default=1, help='Number of epochs of training')
    parser.add_argument('--lr', action='store', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--k', action='store', type=int, default=1, help='Learning rate')
    parser.add_argument('--bs', action='store', type=int, default=1, help='Batch size')
    parser.add_argument('--save', '-s', action='store_true', help="Save or not the DRBN's weights")
    parser.add_argument('--save_path', action='store', type=str, help="Path to save the DRBN's weights")
    parser.add_argument('--fit_cl', action='store_true', help="Train or not the classifier")
    parser.add_argument('--load_w', '-ldw', action='store_true', help="Load or not the DRBN's weights from file")
    parser.add_argument('--w_path', action='store', type=str, help="Path to the DRBN's weights")
    args = parser.parse_args()

    tr_imgs, tr_labels, ts_imgs, ts_labels = load_mnist()
    drbn = DRBN(hl_sizes=args.hl_sizes, v_size=len(tr_imgs[0]), mnist_path='../datasets/MNIST/')
    if args.train:
        drbn.fit(epochs=args.epochs,
                 lr=args.lr,
                 k=args.k,
                 bs=args.bs,
                 save=args.save,
                 save_path=args.save_path,
                 fit_cl=args.fit_cl,
                 save_cl=False,
                 save_cl_path=None)
    else:
        # load weights from file
        drbn.load_weights(args.w_path)
        # train and test classifier
        drbn.fit_classifier(load_boltz_weights=args.load_w, w_path=args.w_path)
        drbn.test_classifier()
        # plot confusion matrix
        drbn.confusion_matrix()
        # plot a reconstruction for each digit
        # indexes = []
        # curr = 0
        # while curr < 10:
        #     for i, label in enumerate(drbn.tr_labels):
        #         if label == curr:
        #             indexes.append(i)
        #             curr += 1
        #             break
        # fig, ax = plt.subplots(2, 10, figsize=(5, 2))
        # for i in range(20):
        #     if i < 10:
        #         ax[0, i].imshow(np.reshape(drbn.tr_imgs[indexes[i]], newshape=(28, 28)))
        #         ax[0, i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        #     else:
        #         v_sample = np.random.binomial(n=1, p=drbn.tr_imgs[indexes[i - 10]], size=len(drbn.tr_imgs[indexes[i - 10]]))
        #         _, h_sample = drbn.forward(v_sample)
        #         h_sample = h_sample[-1]
        #         v_probs, _ = drbn.backward(h_sample)
        #         v_probs = v_probs[0]
        #         ax[1, i - 10].imshow(np.reshape(v_probs, newshape=(28, 28)))
        #         ax[1, i - 10].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        # fig.suptitle("Original images and their reconstructions")
        # fig.show()
        # show reconstructions of specific images
        # new_drbn.show_reconstruction(img=tr_imgs[0])
        # new_drbn.show_reconstruction(img=tr_imgs[1])
