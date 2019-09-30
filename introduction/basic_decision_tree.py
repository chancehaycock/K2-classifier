# Introduction to Machine Learning with sklearn

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read data from excel file
melbourne_data = pd.read_csv('melb_data.csv')

# Filter out the data with missing values. Use the model to predict
# these missing values.
# Note: axis = 0/'index' removes ROWS of missing data
#       axis = 1/'columns' removes COLUMNS of missing data
filtered_melbourne_data = melbourne_data.dropna(axis=0)

# Use dot notation in pandas to access columns.
# Equivalently, use:
#                   y = filtered_melbourne_data['Price']
y = filtered_melbourne_data.Price

# We will use these features from the pandas data table to make our
# estimates. 
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                      'YearBuilt', 'Lattitude', 'Longtitude']

# store the useful variables in a data frame X. Only consists of 7 columns.
# We now use this complete data to make the model.
X = filtered_melbourne_data[melbourne_features]

# From here, we could build our model with the data. However, in reality,
# we would like to know how good our model actually is.

# This is where huge mistakes normally occur. i.e. making predictions from
# the training data, and then comparing the results with the target values
# also in the training data. Models can appear accurate within the training
# data and absolutely awful within a larger sample.

# The train_test_split function splits the data into two pieces. One for
# training, one for validating. We also need to specify a pseudo-random state
# for reproducible results.

# Split data into two pieces.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Initialise a decision tree.
melbourne_model = DecisionTreeRegressor(random_state=0)

# Build a decision tree from the training data.
melbourne_model.fit(train_X, train_y)

# Now the model is built, use it to predict the prices from the
# validation data X
val_predictions = melbourne_model.predict(val_X)

# Use the mean absolute error metric to measure accuracy of the model.
print("Mean Absolute Error is", mean_absolute_error(val_y, val_predictions))

# Misc functions
print("Depth of decision tree is", melbourne_model.get_depth())
print("n_leaves of decision tree is", melbourne_model.get_n_leaves())

# Plot the error
num_points = len(val_predictions)
val_y_array = val_y.to_numpy()
x = np.linspace(0, 500000, num_points)
errors = []
for i in range(num_points):
    errors.append(abs(val_y_array[i] - val_predictions[i]))
plt.scatter(x, errors, c='r', s=0.5)
plt.show()