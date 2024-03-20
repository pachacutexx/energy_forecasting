This project aims to compare the performance of different methodologies to predict the next day energy prices.

It uses the data of the following Kaggle dataset: https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather/data?select=energy_dataset.csv

First, feature engineering is performed. 

Second, we use a SARIMAX model both with and without exogenous variables.

Third, we use a Random Forest Regressor.

Fourth, a series of CLIs are created.

Fifth, an interactive GUI is created.

Both the CLI and GUI follow the same logic:

1. Set the dates for the training data.
2. Select and train the model.
3. Make predictions based on the selected model.
4. Plot the predictions vs the actual values.