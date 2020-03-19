# =============================================================================
# =============================================================================
# som_and_rf_gamma.py
# Created by Chance Haycock February 2020
#
#                        ==============
#                         IMPROVEMENTS
#                        ==============
# Much Larger Training Set - Gamma
#
# =============================================================================
# =============================================================================
from models.model_utilities import *

# =========================================
#      SOM AND RF (K2SC_GAMMA) MODEL
# =========================================

def SOM_and_RF_gamma():

	model_label = 'gamma'
	model_number = sys.argv[1]

	print_model_type("SOM and Random Forest")

	# Import global training data. Contains roughly 100 of each class.
	training_file = "{}/src/models/som_and_rf_{}/train.csv".format(project_dir, model_label)
	df = pd.read_csv(training_file)
	print("Using training file: {}".format(training_file))

	# Features to be tested. Column 0 is epics.
	features = df.drop('class', axis=1).drop('probability', axis=1).columns[1:len(df.columns)-2]

	blank_classifier = RCF(random_state=2, class_weight='balanced')
#	parameters = {'n_estimators':[400, 500, 600],\
#	         'min_samples_split':[3, 4, 5],\
#	              'max_features':[5, 6, 7] }
	parameters = {'n_estimators':[500],\
	         'min_samples_split':[3],\
	              'max_features':[4] }
	# Creates Confusion Matrix, FIMP, Summary Stats on Cross validated model
	evaluate_model(model_label, model_number, blank_classifier, df, parameters, features, in_cv=5, out_cv=5)

	# Do learning curve analysis here

	# Maybe add scores on how well model performs on whole of campaign_4.

	return

def main():
#	make_model_table('gamma')
	SOM_and_RF_gamma()

if __name__ == "__main__":
	main()
