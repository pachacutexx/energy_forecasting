# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from scipy.stats import shapiro
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# Set color palette for color-blind friendly plots
sns.set_palette("colorblind")

# Create the database connection
engine = create_engine('sqlite:///data/processed/database.db')

# Load 'final_features' from the database into a DataFrame
final_features = pd.read_sql('final_features', con=engine)
final_features['time'] = pd.to_datetime(final_features['time'])
final_features.set_index('time', inplace=True)

# Define a function to analyze the time series
def analyze_time_series(series):
    # Plot the series
    plt.figure(figsize=(10, 6))
    plt.plot(series, label='Series')
    plt.title('Time Series')
    plt.show()

    # Plot histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(series, kde=True, color='blue')
    plt.title('Histogram of the Series')
    plt.show()

    # Correlogram (ACF and PACF) with 50 lags
    fig, ax = plt.subplots(2, 1, figsize=(12, 12))
    plot_acf(series, lags=50, ax=ax[0])
    plot_pacf(series, lags=50, method='ols', ax=ax[1])
    plt.show()

    # Augmented Dickey-Fuller test for stationarity
    adf_result = adfuller(series, maxlag=50)
    print(f"ADF Test p-value: {adf_result[1]}")
    if adf_result[1] > 0.05:
        print("Fail to reject the null hypothesis (Ho), the data has a unit root and is non-stationary")
    else:
        print("Reject the null hypothesis (Ho), the data does not have a unit root and is stationary")

    # Ljung-Box test for autocorrelation
    lb_pvalue = sm.stats.acorr_ljungbox(series, lags=[50], return_df=True)
    print(f"Ljung-Box Test p-value: {lb_pvalue.iloc[0, 1]}")
    if lb_pvalue.iloc[0, 1] > 0.05:
        print("Fail to reject the null hypothesis (Ho), the data does not have significant autocorrelation")
    else:
        print("Reject the null hypothesis (Ho), the data has significant autocorrelation")

    # Shapiro-Wilk test for normality
    shapiro_test = shapiro(series)
    print(f"Shapiro-Wilk Test p-value: {shapiro_test[1]}")
    if shapiro_test[1] > 0.05:
        print("Fail to reject the null hypothesis (Ho), the data is normally distributed")
    else:
        print("Reject the null hypothesis (Ho), the data is not normally distributed")

# We analyze the price actual column
print('==========================================')
print('Price Actual time series analysis')
analyze_time_series(final_features['price actual'])
print('==========================================')

# Even though the data is stationary according to the ADF test, the prices tend to be non-stationary.
# We will follow our intuition and take the difference of the actual price.
price_diff = final_features['price actual'].diff().dropna()
print('==========================================')
print('Price Actual time series analysis after differencing')
analyze_time_series(price_diff)
print('==========================================')

# We split the data into X and y
y = final_features['price actual']

# Split the data into training and testing sets
y_train = y.loc['2015-01-01':'2018-12-30']
y_test = y.loc['2018-12-31':'2018-12-31']

# We first train a SARIMA model on the price actual column
# The order of the model is based on the ACF and PACF plots
sarima_model = SARIMAX(y_train,
                       order=(2, 1, 1),              # (p, d, q)
                       seasonal_order=(1, 0, 2, 24)) # (P, D, Q, s)

# Fit the model
sarima_result = sarima_model.fit(disp=False)

# Summary of the model
print(sarima_result.summary())

# Forecast the next 24 hours
y_pred = sarima_result.get_forecast(steps=24).predicted_mean

# Calculate metrics

y_pred = np.array(y_pred)
y_test = np.array(y_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted', linestyle='--')
plt.title('SARIMA 24-hour Ahead Forecast vs Actual')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Output the metrics
print('==========================================')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'MAPE: {mape}')
print('==========================================')