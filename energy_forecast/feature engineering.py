# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings("ignore")

# Set color palette for color-blind friendly plots
sns.set_palette("colorblind")

# Load the datasets
energy_df = pd.read_csv('data/raw/energy_dataset.csv')
weather_df = pd.read_csv('data/raw/weather_features.csv')

# Convert the 'time' columns to datetime
energy_df['time'] = pd.to_datetime(energy_df['time'].astype(str).str.split('+').str[0])
weather_df['dt_iso'] = pd.to_datetime(weather_df['dt_iso'].astype(str).str.split('+').str[0])

# Check for missing values
print(energy_df.info())
print(weather_df.info())

# Drop columns where all the values are missing
energy_df = energy_df.dropna(axis=1, how='all')

# Function to drop columns where all values are the same
def drop_columns_all_same(df):
    return df.loc[:, df.nunique() > 1]

# Drop columns with all the same values from energy_df
energy_df = drop_columns_all_same(energy_df)

# Drop columns with all the same values from weather_df
weather_df = drop_columns_all_same(weather_df)

# Get the unique values of the 'city_name' column
print(weather_df['city_name'].unique())

# Before splitting the weather_df into separate dataframes for each city, we drop columns with the same meaning.
weather_df = weather_df.drop(columns=['weather_icon', 'weather_description', 'weather_main'], axis=1)

# We will check the numerical values of the weather_df dataframe
print(weather_df.describe())

# Plot boxplot of all numerical columns of weather_df
plt.figure(figsize=(12, 6))
weather_df.select_dtypes(include='number').boxplot()
plt.title('Boxplot of Numerical Columns in weather_df')
plt.xticks(rotation=45)
plt.show()

# There are clear outliers in the pressure column, we will truncate them using the IQR.
Q1_pressure = weather_df['pressure'].quantile(0.25)
Q3_pressure = weather_df['pressure'].quantile(0.75)
IQR_pressure = Q3_pressure - Q1_pressure
weather_df.loc[weather_df['pressure'] > (Q3_pressure + 3 * IQR_pressure), 'pressure'] = Q3_pressure + 3 * IQR_pressure
weather_df.loc[weather_df['pressure'] < (Q1_pressure - 3 * IQR_pressure), 'pressure'] = Q1_pressure - 3 * IQR_pressure

# We plot again to better observe the rest of the boxplots
plt.figure(figsize=(12, 6))
weather_df.select_dtypes(include='number').boxplot()
plt.title('Boxplot of Numerical Columns in weather_df')
plt.xticks(rotation=45)
plt.show()

# There seems to be outliers in the 'wind_speed' column, we will truncate them using the IQR too.
Q1_wind_speed = weather_df['wind_speed'].quantile(0.25)
Q3_wind_speed = weather_df['wind_speed'].quantile(0.75)
IQR_wind_speed = Q3_wind_speed - Q1_wind_speed
weather_df.loc[weather_df['wind_speed'] > (Q3_wind_speed + 3 * IQR_wind_speed), 'wind_speed'] = Q3_wind_speed + 3 * IQR_wind_speed
weather_df.loc[weather_df['wind_speed'] < (Q1_wind_speed - 3 * IQR_wind_speed), 'wind_speed'] = Q1_wind_speed - 3 * IQR_wind_speed

# Split weather_df into separate dataframes for each city
city_dfs = {}  # Dictionary to hold the dataframes for each city
for city in weather_df['city_name'].unique():
    city_lower = city.lower()
    city_dfs[city_lower] = weather_df[weather_df['city_name'] == city].copy()

# Create a dataframe for each city
valencia = city_dfs['valencia']
madrid = city_dfs['madrid']
bilbao = city_dfs['bilbao']
barcelona = city_dfs[' barcelona']
seville = city_dfs['seville']

# Drop the 'city_name' column from each city dataframe
valencia.drop(columns=['city_name'], inplace=True)
madrid.drop(columns=['city_name'], inplace=True)
bilbao.drop(columns=['city_name'], inplace=True)
barcelona.drop(columns=['city_name'], inplace=True)
seville.drop(columns=['city_name'], inplace=True)

# Drop duplicates from each city dataframe based on time
valencia.drop_duplicates(subset='dt_iso', inplace=True)
madrid.drop_duplicates(subset='dt_iso', inplace=True)
bilbao.drop_duplicates(subset='dt_iso', inplace=True)
barcelona.drop_duplicates(subset='dt_iso', inplace=True)
seville.drop_duplicates(subset='dt_iso', inplace=True)

# Add suffix to columns to identify the city
valencia.columns = [col + "_val" for col in valencia.columns]
madrid.columns = [col + "_mad" for col in madrid.columns]
bilbao.columns = [col + "_bil" for col in bilbao.columns]
barcelona.columns = [col + "_bar" for col in barcelona.columns]
seville.columns = [col + "_sev" for col in seville.columns]

# Fill null values in energy_df with the previous value
energy_df.fillna(method='ffill', inplace=True)

# Merge the energy_df with the weather data for each city
merged_df = pd.merge(energy_df, valencia, left_on='time', right_on='dt_iso_val', how='left')
merged_df.drop(columns=['dt_iso_val'], axis=1, inplace=True)

merged_df = pd.merge(merged_df, madrid, left_on='time', right_on='dt_iso_mad', how='left')
merged_df.drop(columns=['dt_iso_mad'], axis=1, inplace=True)

merged_df = pd.merge(merged_df, bilbao, left_on='time', right_on='dt_iso_bil', how='left')
merged_df.drop(columns=['dt_iso_bil'], axis=1, inplace=True)

merged_df = pd.merge(merged_df, barcelona, left_on='time', right_on='dt_iso_bar', how='left')
merged_df.drop(columns=['dt_iso_bar'], axis=1, inplace=True)

merged_df = pd.merge(merged_df, seville, left_on='time', right_on='dt_iso_sev', how='left')
merged_df.drop(columns=['dt_iso_sev'], axis=1, inplace=True)

# Create a SQLAlchemy engine
engine = create_engine('sqlite:///data/processed/database.db')

# Ingest the merged_df dataframe into a new table named 'dataset' in the SQLite database
merged_df.to_sql('cleaned', engine, if_exists='replace', index=False)

# Load the data from the 'dataset' table back into a new pandas dataframe
loaded_df = pd.read_sql_table('cleaned', engine)

# Set the 'time' column as the index of the loaded_df dataframe for easier plotting
loaded_df.set_index('time', inplace=True)

# Make sure the shape is as expected and there are no nulls.
print(loaded_df.info())
print(loaded_df.isnull().sum())

# There are similar columns with different timeframes, we will check which one is more inforamtive
# Plot a barchart for the column rain_1h_mad

plt.figure(figsize=(8, 6))
ax = loaded_df['rain_1h_mad'].plot(kind='bar')
plt.title('Rainfall in Madrid (1 hour)')
plt.xlabel('Time')
plt.ylabel('Rainfall (mm)')
tick_frequency = 24 * 60
xticks = range(0, len(loaded_df.index), tick_frequency)
xticklabels = [loaded_df.index[i].strftime('%b-%Y') if i < len(loaded_df.index) else '' for i in xticks]
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=90)
plt.show()

# Plot a barchart for the column rain_1h_mad
plt.figure(figsize=(8, 6))
ax = loaded_df['rain_3h_mad'].plot(kind='bar')
plt.title('Rainfall in Madrid (3 hours)')
plt.xlabel('Time')
plt.ylabel('Rainfall (mm)')
xticks = range(0, len(loaded_df.index), tick_frequency)
xticklabels = [loaded_df.index[i].strftime('%b-%Y') if i < len(loaded_df.index) else '' for i in xticks]
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=90)
plt.show()

# It can be seen that the 1 hour data is more useful than the 3 hour data, we will drop the 3 hour data
# Overall, we consider that more frequent data is more useful
# Drop rain_3h columns
loaded_df.drop(columns=['rain_3h_val', 'rain_3h_mad', 'rain_3h_bil', 'rain_3h_bar', 'rain_3h_sev'], inplace=True)

# Create a column named total_generation that is the sum of all the columns that begin with generation
loaded_df['total_generation'] = loaded_df.filter(like='generation').sum(axis=1)

# Create a column named excess_generation that is the difference between total_generation and total load actual
loaded_df['excess_generation'] = loaded_df['total_generation'] - loaded_df['total load actual']

# Create a column named load_error that is the difference between total load actual and total load forecast
loaded_df['load_error'] = loaded_df['total load actual'] - loaded_df['total load forecast']

# Create 2 columns for months of the year
loaded_df['month'] = loaded_df.index.month
loaded_df['month_2'] = loaded_df.index.month % 12 + 1

# Create 2 columns for day of the week
loaded_df['dow'] = loaded_df.index.dayofweek
loaded_df['dow_2'] = (loaded_df.index.dayofweek + 2) % 7

# Create 2 columns for hour of the day
loaded_df['hour'] = loaded_df.index.hour
loaded_df['hour_2'] = (loaded_df.index.hour - 12) % 24

# New dataframe to be uploaded to the database after feature engineering
final_features = loaded_df.copy()

# Save the final features into another table named 'final_features'
final_features.to_sql('final_features', engine, if_exists='replace', index=True)

print('==========================================')
print('Featuring engineering completed')