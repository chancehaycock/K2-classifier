# =============================================================================
# =============================================================================
# som_and_rf_point4.py
# Created by Chance Haycock February 2020
#
#                        ==============
#                         IMPROVEMENTS
#                        ==============
# All Training Set - Testing performance on population with porbability >= 0.4!
#
# =============================================================================
# =============================================================================
from models.model_utilities import *

model_name = "som_and_rf" 
training_set = 'point4'
# =========================================
#      SOM AND RF (K2SC_ALL) MODEL
# =========================================

def SOM_and_RF_point4():

	model_number = sys.argv[1]

	print_model_type("SOM and Random Forest")

	# Import global training data. Contains roughly 100 of each class.
	training_file = "{}/src/models/som_and_rf_{}/train.csv".format(project_dir, training_set)
	df = pd.read_csv(training_file)
	print("Using training file: {}".format(training_file))

	# Features to be tested
	features = choose_features(df, include_som=True, include_period=True,
                    include_colour=True, include_absmag=True,
                    include_lc_stats=True, include_bin_lc_stats=True)

	blank_classifier = RCF(random_state=2, class_weight='balanced')
#	parameters = {'n_estimators':[400, 500, 600],\
#	         'min_samples_split':[3, 4, 5],\
#	              'max_features':[5, 6, 7] }
	parameters = {'n_estimators':[500],\
	         'min_samples_split':[3],\
	              'max_features':[4] }
	# Creates Confusion Matrix, FIMP, Summary Stats on Cross validated model
	evaluate_model(model_name, training_set, model_number, blank_classifier, df, parameters, features, in_cv=5, out_cv=5)

	# Do learning curve analysis here

	# Maybe add scores on how well model performs on whole of campaign_4.

	return

def main():
#	make_model_table('som_and_rf', 'point4')
	SOM_and_RF_point4()

if __name__ == "__main__":
	main()
