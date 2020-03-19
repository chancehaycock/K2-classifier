# =============================================================================
# =============================================================================
# som_and_etc_delta.py
# Created by Chance Haycock February 2020
#
#                        ==============
#                         IMPROVEMENTS
#                        ==============
# - Hand picked epics
# - Firstly, training set was cut off at threshold of 0.5
# - Each epic was then manually challenged and labels changed accordingly.
#   The result is a coherent training set which as members as follows:

#     New Number of   RRab: 119
#     New Number of     EA: 129
#     New Number of     EB: 159
#     New Number of  DSCUT: 164
#     New Number of   GDOR: 126
#     New Number of  Noise: 10956
#     New Number of OTHPER: 8493
#
# As is stands, there maybe changes to the way the model is evaluated. Perhaps,
# try model cross validation, and hyper-parameter importance.
#
# =============================================================================
# =============================================================================
from models.model_utilities import *
from models.run_on_unseen import *

model_name = "som_and_etc" 
training_set = 'delta'

# =========================================
#      SOM AND ETC (K2SC_DELTA) MODEL
# =========================================

def SOM_and_ETC_delta():

	model_number = sys.argv[1]

	print_model_type("SOM and ETC")

	training_file = "{}/src/models/{}_{}/train.csv".format(project_dir, model_name, training_set)
	df = pd.read_csv(training_file)
	print("Using training file: {}".format(training_file))

	print("Dropping DJA Class column and using 'class' instead")
	df = df.drop('DJA_Class', axis=1)
	print(df)

	# Features to be tested.
	features = choose_features(df, include_som=True, include_period=True,
                    include_colour=True, include_absmag=True,
                    include_lc_stats=True, include_bin_lc_stats=True)

	blank_classifier = RCF(random_state=2, class_weight='balanced')
#	parameters = {'n_estimators':[400, 500, 600],\
#	         'min_samples_split':[3, 4, 5],\
#	              'max_features':[5, 6, 7] }
	parameters = {'n_estimators':[400, 500],\
	         'min_samples_split':[3, 4, 5, 6],\
	              'max_features':[4, 5, 6, 7] }
	num_parameters = len(parameters['n_estimators']) * len(parameters['min_samples_split']) * len(parameters['max_features'])

	# ===========================
	# TWO MAIN EVALUATION OPTIONS
	# ==========================

	# Creates Confusion Matrix, FIMP, Summary Stats on Cross validated model
#	evaluate_model(model_name, training_set, model_number, blank_classifier, df, parameters, num_parameters, features, in_cv=5, out_cv=5)

	# Arbritrarily chosen
	best_parameters = {'n_estimators': 500, 'min_samples_split': 4, 'max_features': 5}

	classifier = ETC(random_state=2, class_weight='balanced', n_estimators=500, min_samples_split=4, max_features=5)

	# SIMPLER MODEL _ PERHAPS BETTER TO GET SCORE OUT OF
	evaluate_model_simpleCV(model_name, training_set, model_number, classifier, df,
                            features, cv=10, umbrella_train_size=500, num_classifiers=50)

	return

def main():
	#make_model_table(model_name, training_set, som_dimension=2)
	SOM_and_ETC_delta()

if __name__ == "__main__":
	main()
