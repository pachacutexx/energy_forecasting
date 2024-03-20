# Importing necessary libraries for data manipulation, statistical modeling, visualization, database interaction, and CLI creation.
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import typer
from typing import Optional
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings to ensure clean output

sns.set_palette("colorblind")  # Set seaborn color palette to 'colorblind' for accessibility

engine = create_engine('sqlite:///data/processed/database.db')  # Create a database connection

final_features = pd.read_sql('final_features', con=engine)
final_features['time'] = pd.to_datetime(final_features['time'])  # Convert 'time' column to datetime
final_features.set_index('time', inplace=True)  # Set the 'time' column as the index

app = typer.Typer()  # Initialize the Typer application to create a CLI

def save_dates_to_file(beg_date, end_date):
    """
    Save the beginning and end dates to a JSON file.

    Parameters:
    - beg_date: The beginning date as a datetime object.
    - end_date: The end date as a datetime object.
    """
    with open("energy_forecast/dates.json", "w") as file:
        json.dump({"beg_date": beg_date.isoformat(), "end_date": end_date.isoformat()}, file)

def load_dates_from_file():
    """
    Load the beginning and end dates from a JSON file.

    Returns:
    A tuple containing the beginning and end dates as datetime objects.
    """
    with open("energy_forecast/dates.json", "r") as file:
        dates = json.load(file)
    return pd.to_datetime(dates["beg_date"]), pd.to_datetime(dates["end_date"])

global_beg_date: Optional[pd.Timestamp] = None
global_end_date: Optional[pd.Timestamp] = None  # Define global variables for storing dates

@app.command()
def set_dates(beg_date: str, end_date: str):
    """
    Set the beginning and end dates for training the SARIMA model.
    """
    global global_beg_date, global_end_date
    min_date = pd.to_datetime("2015-01-01")
    max_date = pd.to_datetime("2018-12-30")

    beg_date = pd.to_datetime(beg_date)
    end_date = pd.to_datetime(end_date)

    if beg_date < min_date:
        typer.echo(f"Error: the beginning date must not be earlier than {min_date.strftime('%Y-%m-%d')}.")
        raise typer.Exit()

    if end_date > max_date:
        typer.echo(f"Error: the end date must not be later than {max_date.strftime('%Y-%m-%d')}.")
        raise typer.Exit()
    
    if beg_date > end_date:
        typer.echo("Error: the beginning date must not be later than the end date.")
        raise typer.Exit()

    global_beg_date = beg_date
    global_end_date = end_date
    save_dates_to_file(global_beg_date, global_end_date)
    typer.echo(f"Beginning date set to {global_beg_date.strftime('%Y-%m-%d')}")
    typer.echo(f"End date set to {global_end_date.strftime('%Y-%m-%d')}")

# Train the SARIMA model

@app.command()
def sarima_model():
    """
    Train the SARIMA model based on the selected beginning and end dates and save the model.
    """
    try:
        # Load dates from the file
        global_beg_date, global_end_date = load_dates_from_file()
    except (FileNotFoundError, json.JSONDecodeError):
        typer.echo("Error: Dates have not been set or the file is corrupted.")
        raise typer.Exit()
    
    # Load time series data
    y = final_features['price actual']
    
    # Slice the data based on selected dates
    y_train = y.loc[global_beg_date:global_end_date]

    # Define SARIMA model configuration
    sarima_model = SARIMAX(y_train,
                           order=(2, 1, 1),              # (p, d, q)
                           seasonal_order=(1, 0, 2, 24), # (P, D, Q, s)
                           enforce_stationarity=False,
                           enforce_invertibility=False)

    # Fit the model
    sarima_result = sarima_model.fit(disp=False)

    # Save the model
    model_filename = "sarima_model.joblib"
    joblib.dump(sarima_result, "energy_forecast/"+model_filename)

    typer.echo(f"Model trained and saved as {model_filename}")

@app.command()
def sarimax_model():
    """
    Train the SARIMAX model based on the selected beginning and end dates and save the model.
    """
    try:
        # Load dates from the file
        global_beg_date, global_end_date = load_dates_from_file()
    except (FileNotFoundError, json.JSONDecodeError):
        typer.echo("Error: Dates have not been set or the file is corrupted.")
        raise typer.Exit()
    
    # Load time series data
    y = final_features['price actual']
    X = final_features.drop(columns=['price actual', 'price day ahead'])
    
    # Slice the data based on selected dates
    y_train = y.loc[global_beg_date:global_end_date]
    X_train = X.loc[global_beg_date:global_end_date]

    # Initialize and fit scalers
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # Fit and transform the features and target
    X_train_scaled = X_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))

    # Define SARIMAX model configuration with scaled data
    sarimax_model = SARIMAX(y_train_scaled.flatten(),
                            exog=X_train_scaled,
                            order=(2, 1, 1),              # (p, d, q)
                            seasonal_order=(1, 0, 2, 24), # (P, D, Q, s)
                            enforce_stationarity=False,
                            enforce_invertibility=False)

    # Fit the model
    sarimax_result = sarimax_model.fit(disp=False)

    # Save the model and the scalers
    model_filename = "sarimax_model.joblib"
    x_scaler_filename = "x_scaler.joblib"
    y_scaler_filename = "y_scaler.joblib"
    joblib.dump(sarimax_result, f"energy_forecast/{model_filename}")
    joblib.dump(X_scaler, f"energy_forecast/{x_scaler_filename}")
    joblib.dump(y_scaler, f"energy_forecast/{y_scaler_filename}")

    typer.echo(f"Model trained and saved as {model_filename}")
    typer.echo(f"X_scaler and Y_scaler saved as {x_scaler_filename} and {y_scaler_filename}, respectively.")

# Random Forest Model
    
@app.command()
def random_forest_model():
    """
    Train the Random Forest Regressor model based on the selected beginning and end dates and save the model.
    """
    try:
        # Load dates from the file
        global_beg_date, global_end_date = load_dates_from_file()
    except (FileNotFoundError, json.JSONDecodeError):
        typer.echo("Error: Dates have not been set or the file is corrupted.")
        raise typer.Exit()

    # Load time series data
    y = final_features['price actual']
    X = final_features.drop(columns=['price actual', 'price day ahead'])
    X_train = X.loc[global_beg_date:global_end_date]
    y_train = y.loc[global_beg_date:global_end_date]

    # Define Random Forest Regressor configuration
    rf_model = RandomForestRegressor(random_state=42,
                                     n_jobs=-1,
                                     bootstrap=False,
                                     max_depth=4,
                                     max_features='sqrt',
                                     n_estimators=50)

    # Fit the model
    rf_model.fit(X_train, y_train)

    # Save the model
    model_filename = "random_forest_model.joblib"
    joblib.dump(rf_model, "energy_forecast/"+model_filename)

    typer.echo(f"Model trained and saved as {model_filename}")

# Prediction

@app.command()
def predict(model: str, steps: int):
    """
    Predict future values using a specified model. The model can be 'sarima', 'sarimax', or 'random_forest'.
    The number of steps to predict must be between 1 and 24.
    """
    # Define the valid models and corresponding file names
    model_files = {
        "sarima": "energy_forecast/sarima_model.joblib",
        "sarimax": "energy_forecast/sarimax_model.joblib",
        "random_forest": "energy_forecast/random_forest_model.joblib"
    }

    # Check if the model string is one of the accepted values
    if model not in model_files:
        typer.echo(f"Error: '{model}' is not a recognized model. Please choose from 'sarima', 'sarimax', or 'random_forest'.")
        raise typer.Exit()

    # Check the number of prediction steps
    if steps < 1 or steps > 24:
        typer.echo("Error: the number of steps must be between 1 and 24.")
        raise typer.Exit()

    # Load the specified model
    model_filename = model_files[model]
    try:
        loaded_model = joblib.load(model_filename)
    except FileNotFoundError:
        typer.echo(f"Error: The model file '{model_filename}' was not found.")
        raise typer.Exit()

    # Predict the future values
    try:
        _, global_end_date = load_dates_from_file()
        global_end_date += pd.Timedelta(days=1)

        if model == 'random_forest':
            X = final_features.drop(columns=['price actual', 'price day ahead'])
            X_test = X.loc[global_end_date:global_end_date + pd.Timedelta(hours=steps-1)]
            y_pred = loaded_model.predict(X_test)
            predictions_df = pd.DataFrame(data=y_pred, columns=['predicted_value'])    
        elif model == 'sarimax':
            # Load the scalers for sarimax model
            x_scaler_filename = "energy_forecast/x_scaler.joblib"
            y_scaler_filename = "energy_forecast/y_scaler.joblib"
            try:
                X_scaler = joblib.load(x_scaler_filename)
                y_scaler = joblib.load(y_scaler_filename)
            except FileNotFoundError as e:
                typer.echo(f"Error: A required file was not found: {e}")
                raise typer.Exit()

            # Load and scale the exogenous features
            X = final_features.drop(columns=['price actual', 'price day ahead'])
            X_test = X.loc[global_end_date:global_end_date + pd.Timedelta(hours=steps-1)]
            X_test_scaled = X_scaler.transform(X_test)

            # Forecast with the scaled exogenous features
            y_pred_scaled = loaded_model.forecast(steps=steps, exog=X_test_scaled)

            # Reverse the scaling of the predictions
            y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            predictions_df = pd.DataFrame(data=y_pred, columns=['predicted_value'])
        else:
            # For SARIMA, proceed without scaling
            y_pred = loaded_model.get_forecast(steps=steps).predicted_mean
            predictions_df = pd.DataFrame(data=y_pred.values, columns=['predicted_value'])    
        # Convert the predictions to a DataFrame with a timestamp index
        predictions_df['timestamp'] = pd.date_range(start=global_end_date, periods=steps, freq='H')

        # Save the DataFrame to the 'predictions' table in 'database.db'
        predictions_df.to_sql('predictions', con=engine, if_exists='replace', index=False)

        typer.echo("Predictions have been saved to the 'predictions' table in 'database.db'.")

    except Exception as e:
        typer.echo(f"An error occurred during prediction or saving the predictions: {e}")
        raise typer.Exit()

# Plot the predictions vs actual data
@app.command()
def plot_predictions():
    """
    Plots predictions versus actual data.
    """    
    # Load 'predictions' from the database into a DataFrame
    df_predictions = pd.read_sql('predictions', con=engine)
    df_predictions['timestamp'] = pd.to_datetime(df_predictions['timestamp'])
    min_timestamp = df_predictions['timestamp'].min()
    max_timestamp = df_predictions['timestamp'].max()
    df_predictions.set_index('timestamp', inplace=True)
    
    # Actual data
    df_actual = final_features['price actual']
    df_actual = final_features.loc[min_timestamp:max_timestamp]

    # Plotting the predictions vs actual data
    plt.figure(figsize=(10, 5))
    plt.plot(df_actual.index, df_actual['price actual'], label='Actual')
    plt.plot(df_predictions.index, df_predictions['predicted_value'], label='Predicted', linestyle='--')
    plt.title('Predictions vs Actual Data')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend()
    plt.tight_layout()
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    app()
