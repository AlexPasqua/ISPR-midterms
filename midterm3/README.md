## Midterm 3 - Assignment 1 (May 2021) - Autoencoders
#### Assignment request:
Train a denoising or a contractive autoencoder on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/): <br>
try out different architectures for the autoencoder, including a single layer autoencoder,
a deep autoencoder with only  layerwise pretraining and a deep autoencoder with fine tuning.<br>
It is up to you to decide how many neurons in each layer and how many layers you want in the deep autoencoder.<br>
Show an accuracy comparison between the different configurations.

Provide a visualization of the encodings of the digits in the highest layer of each configuration, using the t-SNE model
to obtain 2-dimensional projections of the encodings.

Try out what happens if you feed one of the autoencoders with a random noise image and then you apply the iterative
gradient ascent process described in the lecture to see if the reconstruction converges to the data manifold.
