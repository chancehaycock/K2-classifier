# =============================================================================
# =============================================================================
# interim.py
# Created by Chance Haycock December 2019
#
# We first begin with applying the SOM and RF methods to a train/test split of
# campaign 3 and 4 to quanitify the models accuracy.
#
# =============================================================================
# =============================================================================
from model_utilities import *

# =========================================
#             INTERIM MODEL
#    MODEL 0 - ExtraTreesClassifier
# =========================================
def SOM_and_ETC_interim():

	print_model_type("SOM and ETC")

	# Set training split here
	training_split = 0.75

	# Import global training data. Contains roughly 70 of each class.
	training_file = "{}/models/interim.csv".format(project_dir)
	df = pd.read_csv(training_file)

	# Fill empty entries. Maybe try filling these later.
	df = df.fillna(-1)

	# Train test split data as specified with training_split
	train, test = train_test_split(df, stratify=df['class'], random_state=7, train_size=training_split)

	# Features to be tested. Column 0 is epics.
	features = train.columns[1:23]

	# Initialise Classifier
	classifier = ETC(random_state=2, n_estimators=600, class_weight='balanced_subsample', max_features='auto')

	# Fit Classifier
	classifier.fit(train[features], train['class']) 

	# Make Predictions on test data
	test_predictions = classifier.predict(test[features])

	# Get Harsh Classification score of predictions
	score = accuracy_score(test['class'], test_predictions)

	#=================================
	#    Plot Feature Importances
	#=================================
	plot_feature_importances(classifier, features)

	#=================================
	#  PROBABILTIES CONFUSION MATRIX
	#=================================
	plot_confusion_matrix_with_probabilties(classifier, test, features)

	#=================================
	# CLASSIFICATIONS CONFUSION MATRIX
	#=================================
	plot_confusion_matrix_with_classifications(classifier, test, test_predictions)

	#=================================
	#       CLASSIFIER SUMMARY
	#=================================
	print_summary(test, test_predictions, training_file, training_split, score)

	return


# =============================================================================
#                                    MAIN
# =============================================================================

def main():
	print("Main Program Running...\n")
	SOM_and_ETC_interim()

if __name__ == "__main__":
	main()
