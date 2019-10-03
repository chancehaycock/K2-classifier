# Introduction to Machine Learning with sklearn
# Moving on from the previous model, and attempting
# to improve it by implementing a Random Forest from
# the sklean.ensemble package

# Now use RF regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
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

print("The mean absolute error in the random forest model is {}".format(rf_model_mae))

# Now, can we optimise this model (reduce the mae) by playing with the paramter options
# of RandomForestRegressor? By looking in the doc, we find that there are many parameters
# to play with. To begin, we look at the max_leaf_nodes parameter. By default, this is 
# unlimited, and can hence lead to OVERFITTING and UNDERFITTING effects <-- Both bad.

# To get an idea of how this works, we can find our mae for different values of max_leaf_nodes.

# Utility function to find mae on the same data for different values of max_leaf_nodes.
def get_rf_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(random_state=0, max_leaf_nodes=max_leaf_nodes, n_estimators=10)
    model.fit(train_X, train_y)
    val_predict = model.predict(val_X)
    mae = mean_absolute_error(val_predict, val_y)
    return mae

def get_dt_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(random_state=0, max_leaf_nodes=max_leaf_nodes)
    model.fit(train_X, train_y)
    val_predict = model.predict(val_X)
    mae = mean_absolute_error(val_predict, val_y)
    return mae

trials = np.linspace(2, 500, 250)
rf_errors = []
dt_errors = []
for trial in trials:
    rf_errors.append(get_rf_mae(int(trial), train_X, val_X, train_y, val_y))
    dt_errors.append(get_dt_mae(int(trial), train_X, val_X, train_y, val_y))

plt.scatter(trials, rf_errors, c='r', s=0.75, label='Random Forest')
plt.scatter(trials, dt_errors, c='b', s=0.75, label='Decision Tree')
plt.legend()
plt.xlabel("max_leaf_nodes")
plt.ylabel("Mean Abosolute Error")
plt.savefig("dt_errors_vs_rf_errors.png")
plt.close()
