# ISPR Assignments
***Assignments for the course of Intelligent Systems for Pattern Recognition @ University of Pisa***

## Midterm 1 - Assignment 1 (March 2021)
### Autoregressive model
Perform an autoregressive analysis of the “Appliances” column of the dataset which measures the energy consumption of appliances across a period of 4.5 months.<br>
Fit an autoregressive model on the first 3 months of data and estimate performance on the remaining 1.5 months.<br>
Remember to update the autoregressive model as you progress through the 1.5 testing months. For instance, if you have trained the model until time T, use it to predict at time T+1. Then to predict at time T+2 retrain the model using data until time T+1. And so on.<bt>
You might also try and experimenting with less "computationally heavy" retraining schedule (e.g. retrain only "when necessary").<br>
Try out different configurations of the autoregressive model (e.g. experiment with AR models of order 3, 5 and 7). You can use the autoregressive model of your choice (AR, ARMA, ...) and perform data pre-processing operations, if you wish (not compulsory).<br><br>


## Midterm 2 - Assignment 3 (April 2021)
### Restricted Boltzmann Machine
Implement from scratch an RBM and apply it to [MNIST](http://yann.lecun.com/exdb/mnist/).<br>
The RBM should be implemented fully by you (both CD-1 training and inference steps) but you are free to use library functions for the rest (e.g. image loading and management, etc.).
1. Train an RBM with 100 hidden neurons (single layer) on the MNIST data (use the training set split provided by the website).
2. Use the trained RBM to encode all the images using the corresponding activation of the hidden neurons.
3.  Train a simple classifier (e.g. any simple classifier in scikit) to recognize the MNIST digits using as inputs their encoding obtained at step 2. Use the standard training/test split. Show the resulting confusion matrices (training and test) in your presentation.

(Alternative to step 3, optional) Step 3 can as well be realized by placing a softmax layer after the RBM: if you are up for it, feel free to solve the assignment this way.
