# Introduction to Machine Learning with sklearn
# Moving on from the previous model, and attempting
# to improve it by implementing a Random Forest from
# the sklean.ensemble package

# Now use RF regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read training data from excel file
home_data = pd.read_csv('train.csv')

# Target data
y = home_data.SalePrice

# Use these features to predict
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF',
            'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

# Create reduced dataframe of just these columns
X = home_data[features]

# Split the data into training and validating
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Create the random forest Model
rf_model = RandomForestRegressor(random_state=1)

# Fit the model to the training data
rf_model.fit(train_X, train_y)

# Use the model to predict the values of validation input.
val_predict = rf_model.predict(val_X)

# Comapre the actual validation output with the predicted one from the model
rf_model_mae = mean_absolute_error(val_predict, val_y)

print("The mean absolute error in the rnadom forest model is {}".format(rf_model_mae))