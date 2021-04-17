from mlxtend.data import loadlocal_mnist


def load_mnist():
    train_images, train_labels = loadlocal_mnist(images_path='MNIST/train-images-idx3-ubyte',
                                                 labels_path='MNIST/train-labels-idx1-ubyte')
    test_images, test_labels = loadlocal_mnist(images_path='MNIST/t10k-images-idx3-ubyte',
                                               labels_path='MNIST/t10k-labels-idx1-ubyte')
    return train_images, train_labels, test_images, test_labels


if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_mnist()
    print('Dimensions: %s x %s' % (train_images.shape[0], train_images.shape[1]))
