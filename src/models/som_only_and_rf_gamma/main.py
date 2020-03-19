# =============================================================================
# =============================================================================
# som_only_and_rf_gamma.py
# Created by Chance Haycock February 2020
#
# SOM data only passed through RF
#
# =============================================================================
# =============================================================================
from models.model_utilities import *

model_name = "som_only_and_rf" 
training_set = 'gamma'
# =========================================
#      SOM ONLY ON RF (K2SC_BETA) MODEL
# =========================================

def SOM_only():

	model_number = sys.argv[1]

	print_model_type("{} - {} Training Set".format(model_name, training_set))

	# Import global training data. Contains roughly 100 of each class.
	training_file = "{}/src/models/{}_{}/train.csv".format(project_dir, model_name, training_set)
	df = pd.read_csv(training_file)
	print("Using training file: {}".format(training_file))

	# Features to be tested
	features = choose_features(df, include_som=True, include_period=False,
                    include_colour=False, include_absmag=False,
                    include_lc_stats=False, include_bin_lc_stats=False)

	blank_classifier = RCF(random_state=2, class_weight='balanced')
#	parameters = {'n_estimators':[300, 400, 500, 600],\
#	         'min_samples_split':[2, 3, 4, 5, 6],\
#	              'max_features':[4, 5, 6, 7, 8] }
	parameters = {'n_estimators':[300],\
	         'min_samples_split':[3],\
	              'max_features':[4] }
	# Creates Confusion Matrix, FIMP, Summary Stats on Cross validated model
	evaluate_model(model_name, training_set, model_number, blank_classifier, df, parameters, features, in_cv=5, out_cv=5)

	# Do learning curve analysis here

	# Maybe add scores on how well model performs on whole of campaign_4.
	# XXX ADD Random/stratified campaign 4 split example with best params?

	return

def main():

#	make_model_table(model_name, training_set)
	SOM_only()

if __name__ == "__main__":
	main()
