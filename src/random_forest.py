from kepler_data_utilities import *
from sklearn.ensemble import RandomForestClassifier
# Get Importance of Features from RF classifier. Initially this is set up to
# test the importance of our basic features.

campaign_num=3

# Import csv
df = pd.read_csv('{}/tables/campaign_{}_master_table.csv'\
                 .format(px402_dir, campaign_num), 'r', delimiter=',')
# Drop irrelevant rows.
df = df.dropna()
# Useless periods
df = df[(df['Period_1'] != 20.0) & (df['Period_2'] != 20.0)]

# Class Labels
y = [type.strip() for type in df["Class"].values]

# Training Data
X = df.drop("epic_number", axis=1).drop("Class", axis=1)

print(X)
# RF Object with 100 estimators
classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(X, y)

# Standard Deviation of the feature importances.
std = np.std([tree.feature_importances_ for tree in classifier.estimators_],
              axis=0)
importance = classifier.feature_importances_
indices = np.argsort(importance)
plt.barh(range(X.shape[1]), importance[indices], color="r", xerr=std[indices],
         align="center")
features = [label for label in X.columns]
ordered_features = [features[indices[i]] for i in range(len(indices))]
plt.yticks(range(len(ordered_features)), ordered_features)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

