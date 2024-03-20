## Group 5 - Energy Price Forecasting

This project aims to compare the performance of different methodologies to predict the next-day energy prices.

It uses the data of the following Kaggle dataset: https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather/data?select=energy_dataset.csv

The dataset has generation, load, and prices of energy at an hourly frequency. In addition, it contains weather information of 5 major Spanish cities. 

First, feature engineering is performed. At a high level the following is done:

1. Fill null values with the last observations.
2. Drop duplicates.
3. For the weather dataset, create columns for each city so that it has the same rows as the energy dataset.
4. Aggregate columns.
5. Drop uninformative columns.
6. Create time-related dummies to account for the time cycle.

Second, we use a SARIMAX model both with and without exogenous variables. The SARIMAX model uses the both the past of the energy price and the present of exogenous variables. The SARIMA model, without exogenous variables, only uses the past of the energy price. The order of both models is the same and it was selected using the ACF and PACF criteria.

Third, we use a Random Forest Regressor. This is an ensemble method that combines several trees to reduce the variance and overfitting associated to decision trees. We tuned the maximum depth and number of estimators hyperparameters before choosing a final model.

Fourth, a series of CLIs are created.

Fifth, an interactive GUI is created.

Both the CLI and GUI follow the same logic:

1. Set the dates for the training data.
2. Select and train the model.
3. Make predictions based on the selected model.
4. Plot the predictions vs the actual values.

## Navigating the project

The raw and processed data can be found within the data file.

The Python files to perform feature engineering, model selection, and the creation of the CLIs and the GUI are inside the energy_forecast file.

## CLI usage instructions (cli.py)

This Python script defines a Command Line Interface (CLI) application for forecasting energy prices using machine learning models. It leverages the Typer library to facilitate interaction through the command line, encompassing functionalities such as setting date ranges, training models, making predictions, and plotting results. Here's a detailed guide on each component and instructions for their use:

### Setting Dates
- **Functionality**: Allows users to define the beginning and end dates for the dataset intended for model training. These dates are preserved in a JSON file.
- **Usage**: Execute `python script_name.py set-dates --beg-date YYYY-MM-DD --end-date YYYY-MM-DD` in the command line, substituting `script_name.py` with your actual script name and providing the desired beginning and end dates.

### Training Models
- **Functionality**: Facilitates the training of three distinct models: SARIMA, SARIMAX, and Random Forest, utilizing data from the specified date range. The trained model is subsequently saved to a file.
- **Usage**: Post date setting, initiate model training by executing one of the following commands based on the model choice:
  - For SARIMA: `python script_name.py sarima-model`
  - For SARIMAX: `python script_name.py sarimax-model`
  - For Random Forest: `python script_name.py random-forest-model`

### Making Predictions
- **Functionality**: Users can generate predictions using any of the pre-trained models. The prediction process is based on a specified number of steps (hours) and the results are stored in a database.
- **Usage**: Run `python script_name.py predict --model model_name --steps N`, where `model_name` is one of `sarima`, `sarimax`, or `random_forest`, and `N` represents the number of steps (hours) for prediction.

### Plotting Predictions
- **Functionality**: This feature plots the predicted values alongside the actual data for the designated prediction interval.
- **Usage**: To visualize the predictions against actual data, execute `python script_name.py plot-predictions`.

### General Instructions:
1. **Date Range Setting**: It's imperative to specify the beginning and end dates for the dataset before proceeding with model training.
2. **Model Training**: Select and train a model of your choice, ensuring it encompasses the period of interest for forecasting.
3. **Prediction Generation**: Post model training, utilize the model for forecasting over a specified number of future steps.
4. **Prediction Visualization**: Lastly, the prediction outcomes can be visualized by plotting them in comparison to actual data.

## Streamlit GUI usage instructions

This Streamlit application is a user-friendly graphical interface designed for forecasting energy prices. Users can easily select date ranges, train machine learning models, make predictions, and visualize outcomes. Here's a step-by-step guide on how to use the application:

### Setting Dates for Model Training
1. **Navigate to "Set Dates for Model Training"**: This section contains two date input fields for selecting your date range.
2. **Choose the Beginning Date**: Click on the first date input to open a calendar and select the start date for your data.
3. **Choose the End Date**: Click on the second date input to select the end date for your data range.
4. **Save Dates**: Click the "Save Dates" button. If the dates are outside the valid range or the beginning date is after the end date, you'll see an error message.

### Training a Model
1. **Proceed to the "Train Model" Section**: You'll find a dropdown menu for selecting the model type (SARIMA, SARIMAX, or Random Forest).
2. **Select Model Type**: Use the dropdown to choose the model you want to train.
3. **Initiate Model Training**: Press the "Train Model" button. A loading spinner will indicate the training process, followed by a success message upon completion.

### Making Predictions
1. **Move to the "Predict Future Values" Section**: This part allows you to specify the forecast length in steps (hours).
2. **Input Steps for Prediction**: Adjust the number input to set how many steps ahead you want to forecast, up to 24 steps.
3. **Start Prediction**: Hit the "Predict" button. If there's an issue with the model or model type, an error message will be shown.

### Visualizing Predictions
1. **Go to "Plot Predictions vs Actual Data" Section**: This is where you can compare your forecasts against actual historical data.
2. **Generate Plot**: Click the "Plot" button. If the date range is not set correctly, an error message will prompt you to adjust it.

### Additional Information
- **Date Saving/Loading**: The application handles the date range by saving it to a file, which is then used across various operations to ensure consistency.
- **Model and Scaler Files**: In the case of the SARIMAX model, the application also manages data scalers, saving them along with the model to ensure predictions follow the same data preprocessing as the training phase.
- **Displaying Predictions**: Post-prediction, the application shows a DataFrame with forecasted values and calculates the Mean Absolute Error (MAE) against actual data.
- **Database Storage**: Predictions are stored in a database, facilitating their retrieval for visualization without needing to re-run predictions.

## Installation Guide for Project Dependencies

This project uses Poetry for dependency management. To install the dependencies, follow these steps:

1. **Install Poetry**: If you don't already have Poetry installed, follow the instructions on the [Poetry website](https://python-poetry.org/docs/#installation).

2. **Clone the Repository**: Clone the project repository to your local machine using Git: git clone <repository-url>

Replace `<repository-url>` with the URL of the project's Git repository.

3. **Navigate to the Project Directory**: Change into the project's root directory where the `pyproject.toml` and `poetry.lock` files are located: cd path/to/project

Replace `path/to/project` with the actual path to your project directory.

4. **Install Dependencies**: Run the following command to install the project dependencies: poetry-install

This command will create a virtual environment and install all the necessary packages specified in `pyproject.toml` and `poetry.lock` files, ensuring that you have the right package versions to run the project successfully.



