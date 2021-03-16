# ISPR Assignments
### Assignments for the course of Intelligent Systems for Pattern Recognition @ University of Pisa

## Midterm 1 - Assignment 1 (March 2021)
### Autoregressive model
Perform an autoregressive analysis of the “Appliances” column of the dataset which measures the energy consumption of appliances across a period of 4.5 months. Fit an autoregressive model on the first 3 months of data and estimate performance on the remaining 1.5 months. Remember to update the autoregressive model as you progress through the 1.5 testing months. For instance, if you have trained the model until time T, use it to predict at time T+1. Then to predict at time T+2 retrain the model using data until time T+1. And so on. You might also try and experimenting with less "computationally heavy" retraining schedule (e.g. retrain only "when necessary").  Try out different configurations of the autoregressive model (e.g. experiment with AR models of order 3, 5 and 7). You can use the autoregressive model of your choice (AR, ARMA, ...) and perform data pre-processing operations, if you wish (not compulsory).
