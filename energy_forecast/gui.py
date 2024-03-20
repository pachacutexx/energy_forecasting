import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# Set color palette for color-blind friendly plots
sns.set_palette("colorblind")

# Variables to store the trained model and dates
trained_model = None
beg_date = None
end_date = None

# Load 'final_features' from the database into a DataFrame
@st.cache_data
def load_data():
    engine = create_engine('sqlite:///data/processed/database.db')
    final_features = pd.read_sql('final_features', con=engine)
    final_features['time'] = pd.to_datetime(final_features['time'])
    final_features.set_index('time', inplace=True)
    return final_features

final_features = load_data()

# Helper functions for date management
def save_dates_to_file(beg_date, end_date):
    with open("energy_forecast/dates.json", "w") as file:
        json.dump({"beg_date": beg_date.isoformat(), "end_date": end_date.isoformat()}, file)

def load_dates_from_file():
    with open("energy_forecast/dates.json", "r") as file:
        dates = json.load(file)
        return pd.to_datetime(dates["beg_date"]), pd.to_datetime(dates["end_date"])

# Streamlit UI
st.title('Energy Price - Forecast Application')

# Date selection
st.header('Set Dates for Model Training')
col1, col2 = st.columns(2)
with col1:
    beg_date = st.date_input("Beginning Date", pd.to_datetime("2015-01-01").date())
with col2:
    end_date = st.date_input("End Date", pd.to_datetime("2015-06-30").date())

if st.button('Save Dates'):
    valid_beg_date = pd.to_datetime("2015-01-01").date()
    valid_end_date = pd.to_datetime("2018-12-30").date()
    
    if beg_date < valid_beg_date or end_date > valid_end_date or beg_date > end_date:
        st.error("Please ensure dates are within the valid range and the beginning date is before the end date.")
    else:
        # Assuming save_dates_to_file(beg_date, end_date) is defined elsewhere
        save_dates_to_file(beg_date, end_date)
        st.success(f"Dates set to: {beg_date} - {end_date}")

# Function to train the SARIMA model
def train_sarima(beg_date, end_date):
    # Slice the data based on selected dates
    y = final_features['price actual']
    y_train = y.loc[beg_date:end_date]

    # Define SARIMA model configuration
    sarima_model = SARIMAX(y_train,
                            order=(2, 1, 1),              # (p, d, q)
                            seasonal_order=(1, 0, 2, 24), # (P, D, Q, s)
                            enforce_stationarity=False,
                            enforce_invertibility=False)

    # Fit the model
    sarima_result = sarima_model.fit(disp=False)
    
    return sarima_result

# Function to train the SARIMAX model
def train_sarimax(beg_date, end_date):
    # Slice the data based on selected dates
    y = final_features['price actual']
    X = final_features.drop(columns=['price actual', 'price day ahead'])
    
    y_train = y.loc[beg_date:end_date]
    X_train = X.loc[beg_date:end_date]

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
                            enforce_invertibility=True)

    # Fit the model
    sarimax_result = sarimax_model.fit(disp=False)

    return sarimax_result, X_scaler, y_scaler


def train_random_forest(beg_date, end_date):
    y = final_features['price actual']
    X = final_features.drop(columns=['price actual', 'price day ahead'])
    # Slice the data based on selected dates
    X_train = X.loc[beg_date:end_date]
    y_train = y.loc[beg_date:end_date]

    # Define Random Forest model configuration
    rf_model = RandomForestRegressor(random_state=42,
                                     n_jobs=-1,
                                     bootstrap=False,
                                     max_depth=4,
                                     max_features='sqrt',
                                     n_estimators=50)

    # Fit the model
    rf_model.fit(X_train, y_train)

    return rf_model


# Streamlit UI for Model Training
st.header('Train Model')
model_type = st.selectbox('Select Model Type', ['sarima', 'sarimax', 'random_forest'])

if st.button('Train Model'):
    if model_type == 'sarima':
        # Assume beg_date and end_date have been set using st.date_input() earlier in the code
        if beg_date and end_date:
            # Call the function to train the SARIMA model
            with st.spinner(f"Training {model_type} model... Please wait."):
                sarima_result = train_sarima(beg_date, end_date)
                model_filename = "sarima_model.joblib"
                joblib.dump(sarima_result, f"energy_forecast/{model_filename}")
                st.success(f"Model trained and saved as {model_filename}")
                st.session_state['trained_model'] = sarima_result
        else:
            st.error("Please select both beginning and end dates before training the model.")
    
    elif model_type == 'sarimax':
        if beg_date and end_date:
            # Call the function to train the SARIMAX model
            with st.spinner(f"Training {model_type} model... Please wait."):
                sarimax_result, X_scaler, y_scaler = train_sarimax(beg_date, end_date)
                
                # Save the model and scalers
                model_filename = "energy_forecast/sarimax_model.joblib"
                x_scaler_filename = "energy_forecast/x_scaler.joblib"
                y_scaler_filename = "energy_forecast/y_scaler.joblib"
                
                joblib.dump(sarimax_result, model_filename)
                joblib.dump(X_scaler, x_scaler_filename)
                joblib.dump(y_scaler, y_scaler_filename)

                st.session_state['trained_model'] = sarimax_result
                st.session_state['x_scaler'] = X_scaler
                st.session_state['y_scaler'] = y_scaler

                st.success(f"SARIMAX model trained and saved as {model_filename}")
        else:
            st.error("Please select both beginning and end dates before training the model.") 

    elif model_type == 'random_forest':
        if beg_date and end_date:
            with st.spinner(f"Training {model_type} model... Please wait."):
                rf_result = train_random_forest(beg_date, end_date)
                model_filename = "random_forest_model.joblib"
                joblib.dump(rf_result, f"energy_forecast/{model_filename}")
                st.success(f"Model trained and saved as {model_filename}")
                st.session_state['trained_model'] = rf_result        

    else:
                st.error(f"Unsupported model type selected: {model_type}")

# Prediction
st.header('Predict Future Values')
steps = st.number_input('Steps to Predict', min_value=1, max_value=24, value=24)

# Update model type selection based on user input
st.session_state['model_type'] = model_type

if st.button('Predict'):

    model = st.session_state.get('trained_model')
    model_type = st.session_state.get('model_type')

    # Select the appropriate period
    prediction_end_date = end_date + pd.Timedelta(days=1)
    prediction_index = pd.date_range(start=prediction_end_date, periods=steps, freq='H')

    if model_type in ['sarima', 'sarimax'] and model is not None:
        with st.spinner(f"Predicting the next {steps} steps..."):   
            
            if model_type == 'sarimax':
                # Load the scalers
                X_scaler = st.session_state.get('x_scaler')
                y_scaler = st.session_state.get('y_scaler')  

                X_test = final_features.drop(columns=['price actual', 'price day ahead'])
                X_test = X_test.loc[prediction_index]
                X_test_scaled = X_scaler.transform(X_test)
                # Make predictions
                y_pred_scaled = model.forecast(steps=steps, exog=X_test_scaled)

                # Inverse transform the predictions to original scale
                y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

            else:
                y_pred = model.get_forecast(steps=steps).predicted_mean
                y_pred = y_pred.values

    elif model_type == 'random_forest':
                X = final_features.drop(columns=['price actual', 'price day ahead'])
                X_test = X.loc[prediction_index]
                y_pred = model.predict(X_test)
    
    else:
        st.error("Model is not trained or unsupported model type selected.")
    
    predictions_df = pd.DataFrame({'predicted_value': y_pred}, index=prediction_index)
    predictions_df.reset_index(inplace=True)
    predictions_df.rename(columns={'index': 'timestamp'}, inplace=True)

    y = final_features['price actual']
    y_test = y.loc[prediction_index]
    # MAE calculation (uncomment and adjust when you have true values)
    mae = mean_absolute_error(y_test, y_pred)
    mae = np.round(mae, 2)
    st.write(f"Mean Absolute Error: {mae}")

    # Now, save the DataFrame to the 'predictions' table in 'database.db'
    engine = create_engine('sqlite:///data/processed/database.db')
    predictions_df.to_sql('predictions', con=engine, if_exists='replace', index=False)

    st.success(f"Prediction completed for the next {steps} steps.")
    st.write(predictions_df)
    # Save the trained model to the session state
    st.session_state['trained_model'] = trained_model

# Plotting
st.header('Plot Predictions vs Actual Data')
if st.button('Plot'):
    # Ensure that beg_date and end_date are set
    if beg_date is not None and end_date is not None:
        with st.spinner("Plotting... Please wait."):
            # Create a SQL engine
            engine = create_engine('sqlite:///data/processed/database.db')
            
            # Read the predictions from the 'predictions' table in 'database.db'
            predictions_df = pd.read_sql('predictions', con=engine)
            predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
            min_date = predictions_df['timestamp'].min()
            max_date = predictions_df['timestamp'].max()
            predictions_df.set_index('timestamp', inplace=True)

            # Fetch actual values for the same timestamps
            actual_data = final_features['price actual']
            actual_data = actual_data.loc[min_date:max_date]

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(actual_data.index, actual_data, label='Actual')
            ax.plot(predictions_df.index, predictions_df['predicted_value'], label='Predicted', linestyle='--')
            ax.set_title('Predictions vs Actual Data')
            ax.set_xlabel('Time')
            ax.set_ylabel('Price Actual')
            ax.legend()
            st.pyplot(fig)
    else:
        st.error("Dates are not set.")
