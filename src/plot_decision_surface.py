from kepler_data_utilities import *
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import brier_score_loss
from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
plt.style.use('seaborn-white')

def label_map(y):
    numbers = []
    for label in y:
        if label == 'RRab':
            numbers.append(0)
        if label == 'EA':
            numbers.append(1)
        if label == 'EB':
            numbers.append(2)
        if label == 'GDOR':
            numbers.append(3)
        if label == 'DSCUT':
            numbers.append(4)
        if label == 'Noise':
            numbers.append(5)
        if label == 'OTHPER':
            numbers.append(6)
    return np.array(numbers)


# Parameters
n_classes = 3
n_estimators = 100
#cmap = plt.cm.YlGnBu
cmap = plt.cm.Accent
plot_step = 0.02  # fine step width for decision surface contours
plot_step_coarser = 0.5  # step widths for coarse classifier guesses
RANDOM_SEED = 13  # fix the seed on each iteration

# Load data
training_file = "{}/training_sets/c34_clean_1_RF.csv".format(project_dir)
df = pd.read_csv(training_file)

# Fill empty entries. Maybe try filling these later.
df = df.fillna(-1)

features = df.columns[1:23]
plot_idx = 1
models = [ExtraTreesClassifier(n_estimators=n_estimators)]

for pair in [['Period_1', 'skew']]:
    for model in models:
        # We only take the two corresponding features
        
        X = df[features]
        X = X[pair]
        y = label_map(df['class'])

        # Shuffle
        idx = np.arange(X.shape[0])
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(idx)
        X = X.iloc[idx]
        y = y[idx]

        # Standardize
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / std

        # Train
        model.fit(X, y)

        scores = model.score(X, y)
        # Create a title for each column and the console by using str() and
        # slicing away useless parts of the string
        model_title = str(type(model)).split(
            ".")[-1][:-2][:-len("Classifier")]

        model_details = model_title
        if hasattr(model, "estimators_"):
            model_details += " with {} estimators".format(
                len(model.estimators_))
        print(model_details + " with features", pair,
              "has a score of", scores)

        if plot_idx <= len(models):
            # Add a title at the top of each column
            plt.title(model_title, fontsize=9)

        # Now plot the decision boundary using a fine mesh as input to a
        # filled contour plot
        x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
        y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        # Plot either a single DecisionTreeClassifier or alpha blend the
        # decision surfaces of the ensemble of classifiers
        if isinstance(model, DecisionTreeClassifier):
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=cmap)
        else:
            # Choose alpha blend level with respect to the number
            # of estimators
            # that are in use (noting that AdaBoost can use fewer estimators
            # than its maximum if it achieves a good enough fit early on)
            estimator_alpha = 1.0 / len(model.estimators_)
            for tree in model.estimators_:
                Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

        # Build a coarser grid to plot a set of ensemble classifications
        # to show how these are different to what we see in the decision
        # surfaces. These points are regularly space and do not have a
        # black outline
        xx_coarser, yy_coarser = np.meshgrid(
            np.arange(x_min, x_max, plot_step_coarser),
            np.arange(y_min, y_max, plot_step_coarser))
        Z_points_coarser = model.predict(np.c_[xx_coarser.ravel(),
                                         yy_coarser.ravel()]
                                         ).reshape(xx_coarser.shape)
        #cs_points = plt.scatter(xx_coarser, yy_coarser, s=20,
         #                       c=Z_points_coarser, cmap=cmap,
          #                      edgecolors="none")

        # Plot the training points, these are clustered together and have a
        # black outline

        plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y,
                    cmap=cmap,
                    edgecolor='k', s=20)
        plot_idx += 1  # move on to the next plot in sequence

plt.suptitle("Classifiers on feature subsets of Campaign 3 and 4 Training Data", fontsize=10)
plt.axis("tight")
plt.tight_layout(h_pad=0.2, w_pad=0.2, pad=2.5)
plt.show()
