# =============================================================================
# =============================================================================
# som_and_rf_beta.py
# Created by Chance Haycock January 2020
#
#                        ==============
#                         IMPROVEMENTS
#                        ==============
#
# - Bigger Training Set than Alpha
# - Lightcurves now folded on ALL periods (even if == 20.0)
# - All lightcurves hence have 64 bin values and lightcurve statistics
# - Missing bin values have now been linearly interpolated
# - Point-point statistics now calculated on median DIVIDED flux
# - Colour included
#
# =============================================================================
# =============================================================================
from models.model_utilities import *

# =========================================
#      SOM AND RF (K2SC_BETA) MODEL
# =========================================

def SOM_and_RF_beta():

	model_label = 'beta'
	model_number = sys.argv[1]

	print_model_type("SOM and Random Forest")

	# Import global training data. Contains roughly 100 of each class.
	training_file = "{}/src/models/som_and_rf_beta/train.csv".format(project_dir)
	df = pd.read_csv(training_file)
	print("Using training file: {}".format(training_file))

	# Features to be tested. Column 0 is epics.
	features = df.drop('class', axis=1).drop('probability', axis=1).columns[1:len(df.columns)-2]

	blank_classifier = RCF(random_state=2, class_weight='balanced')
	parameters = {'n_estimators':[300, 400, 500, 600],\
	         'min_samples_split':[2, 3, 4, 5, 6],\
	              'max_features':[4, 5, 6, 7, 8] }
#	parameters = {'n_estimators':[300],\
#	         'min_samples_split':[3],\
#	              'max_features':[4] }
	# Creates Confusion Matrix, FIMP, Summary Stats on Cross validated model
	evaluate_model(model_label, model_number, blank_classifier, df, parameters, features, in_cv=5, out_cv=5)

	# Do learning curve analysis here

	return


# =============================================================================
#                                    MAIN
# =============================================================================

def main():
#	make_model_table()
	SOM_and_RF_beta()

if __name__ == "__main__":
	main()
