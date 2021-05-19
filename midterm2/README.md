## Midterm 2 - Assignment 3 (April 2021) - Restricted Boltzmann Machine
#### Assignment request:
Implement from scratch an RBM and apply it to [MNIST](http://yann.lecun.com/exdb/mnist/). <br>
The RBM should be implemented fully by you (both CD-1 training and inference steps) but you are free to use library functions for the rest (e.g. image loading and management, etc.).
1. Train an RBM with 100 hidden neurons (single layer) on the MNIST data (use the training set split provided by the website).
2. Use the trained RBM to encode all the images using the corresponding activation of the hidden neurons.
3.  Train a simple classifier (e.g. any simple classifier in scikit) to recognize the MNIST digits using as inputs their encoding obtained at step 2. Use the standard training/test split. Show the resulting confusion matrices (training and test) in your presentation.

(Alternative to step 3, optional) Step 3 can as well be realized by placing a softmax layer after the RBM: if you are up for it, feel free to solve the assignment this way.

#### Implementation:
- **Restricted Boltzmann Machine** implemented in [RBM.py](RBM.py)
- **Deep Restricted Boltzmann Network** ([reference](https://arxiv.org/pdf/1611.07917.pdf)):
    - Implemented in [DRBN.py](DRBN.py)
    - All the layers are trained jointly with (persistent) contrastive divergence ((P)CD)
    - PCD implemented using a single Gibbs sampling chain

#### References:
1. Hu, Gao, Ma. _Deep Restricted Boltzmann Networks_. 2016 ([link](https://arxiv.org/pdf/1611.07917.pdf))
2. Tieleman, _Training Restricted Boltzmann Machines using Approximations to the Likelihood Gradient_. 2008 ([link](http://icml2008.cs.helsinki.fi/papers/638.pdf))