# Taxi-Trip-Duration-Estimation
Regression for NYC Taxi Trip Duration with Keras based on pick-up coordinates, drop-off coordinates, pick-up time and the number of passengers. RMSLE score of 0.45 (Top 55% of the leaderboard) https://www.kaggle.com/c/nyc-taxi-trip-duration

Simple neural network architecture on Keras of 2 dense layers with size 50, with PReLu activation function, dropouts and layer normalizations. Reduction of the learning rate after 10 epochs without progress. The loss function is MSLE.

The closer a data point lies to the diagonal line, the better the prediction.
(NOTE: The skewness here is due to the MSLE (as opposed to MSE) loss function)

![Test Image 1](https://github.com/fallintoplace/Taxi-Trip-Duration-Calculation/blob/master/prediction_graph.png)
![Test Image 2](https://github.com/fallintoplace/Taxi-Trip-Duration-Calculation/blob/master/loss_graph.png)
