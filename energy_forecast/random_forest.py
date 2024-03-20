# Importing the libraries
import pandas as pd
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
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

X = final_features.drop(columns=['price actual', 'price day ahead'])
y = final_features['price actual']

X_train = X.loc['2015-01-01':'2018-12-30']
y_train = y.loc['2015-01-01':'2018-12-30']
X_test = X.loc['2018-12-31':'2018-12-31']
y_test = y.loc['2018-12-31':'2018-12-31']

# Train a Random Forest model using Grid Search Cross Validation

# Define the hyperparameter grid
param_grid = {'n_estimators': [50, 100, 150, 200],
                       'max_depth': [1, 2, 3, 4, 5],
                       'max_features': ['sqrt', 'log2']}

# Instantiate the Random Forest Regressor
estimator = RandomForestRegressor(random_state=42, n_jobs=-1, bootstrap=False)

# TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Instantiate the GridSearchCV object
model = GridSearchCV(estimator, param_grid, cv=tscv, n_jobs=-1, verbose=0)

# Fit the model
model.fit(X_train, y_train)

# Get the best parameters
best_params = model.best_params_
print(f"Best parameters: {best_params}")

# Predict the test set
y_pred = model.predict(X_test)

# Calculate the Mean Absolute Error
mae_train = mean_absolute_error(y_train, model.predict(X_train))
mae_test = mean_absolute_error(y_test, y_pred)

print(f"Train MAE: {mae_train:.2f}")
print(f"Test MAE: {mae_test:.2f}")


                       