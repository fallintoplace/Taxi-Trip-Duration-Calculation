# Taxi-Trip-Duration-Calculation
Regression for NYC Taxi Trip Duration with Keras based on coordinates, pick-up time and the number of passengers. RMSLE score of 0.46 (Top 60% of the leaderboard) https://www.kaggle.com/c/nyc-taxi-trip-duration

Simple neural network architecture on Keras of 2 dense layers with size 50, with PReLu activation function, dropouts and layer normalizations. Early stopping after 20 epochs. The loss function is MSLE.

![Test Image 1](https://github.com/fallintoplace/Taxi-Trip-Duration-Calculation/blob/master/prediction_graph.png)
![Test Image 2](https://github.com/fallintoplace/Taxi-Trip-Duration-Calculation/blob/master/loss_graph.png)
