# =============================================================================
# =============================================================================
# som_and_rf_delta.py
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

model_name = "som_and_rf" 
training_set = 'delta'

# =========================================
#      SOM AND RF (K2SC_DELTA) MODEL
# =========================================

def SOM_and_RF_delta():

	model_number = sys.argv[1]

	print_model_type("SOM and Random Forest")

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

	classifier = RCF(random_state=2, class_weight='balanced', n_estimators=500, min_samples_split=4, max_features=5)

	# SIMPLER MODEL _ PERHAPS BETTER TO GET SCORE OUT OF
	evaluate_model_simpleCV(model_name, training_set, model_number, classifier, df,
                            features, cv=10, umbrella_train_size=500, num_classifiers=50)



#	# =================
#	#   RUN ON UNSEEN
#	# =================
#	max_features = best_parameters['max_features']
#	min_samples_split = best_parameters['min_samples_split']
#	n_estimators = best_parameters['n_estimators']
#	umbrella_train_size = 500
#	num_classifiers = 50
#	num_unknown = 101112
#	num_classes = 7
#	predictions_array = np.zeros(shape=(num_classifiers, num_unknown, num_classes))
#	# Open test table
#	test_df = pd.read_csv("{}/src/models/{}_{}/test_with_camp.csv".format(project_dir, model_name, training_set))
#	epics = test_df['epic_number']
#	return 
#	for i in range(num_classifiers):
#		final_classifier = RCF(n_estimators=n_estimators, min_samples_split=min_samples_split,
#		                       max_features=max_features, random_state=2, class_weight='balanced')
#
#		# Quick downsampling of larger sized classes
#		print("Downsampling OTHPER and Noise to {} members using df.sample(random_state={}).".format(umbrella_train_size, i))
#		if len(df[df['class'] == 'OTHPER']) > umbrella_train_size:
#			df_without_umb = df[(df['class'] != 'OTHPER') & (df['class'] != 'Noise')]
#			othper_sample = df[df['class'] == 'OTHPER'].sample(n=umbrella_train_size, random_state=i)
#			noise_sample = df[df['class'] == 'Noise'].sample(n=umbrella_train_size, random_state=i)
#			train_df = df_without_umb.append(othper_sample, ignore_index=True).append(noise_sample, ignore_index=True)
#			train_df = train_df.sample(frac=1)
#		print('Down-Sampled Training Set:')
#		print(train_df)
#
#		final_classifier.fit(train_df[features], train_df['class'])
#		predictions = run_on_unseen(final_classifier, model_name, training_set, test_df, features)
#		print('Predictions:')
#		print(predictions)
#		predictions_array[i] = predictions
#
#	columns = final_classifier.classes_
#	final_predictions_df = pd.DataFrame(np.mean(predictions_array, axis=0), columns=columns)
#	final_predictions_df['epic_number'] = test_df['epic_number']
#	final_predictions_df['Campaign'] = test_df['Campaign']
#	variables = ['RRab', 'EA', 'EB', 'GDOR', 'DSCUT', 'OTHPER', 'Noise']
#	final_predictions_df['Class'] = final_predictions_df[variables].idxmax(axis=1)
#	final_predictions_df = final_predictions_df[['epic_number', 'Campaign', 'DSCUT', 'EA', 'EB', 'GDOR', 'Noise', 'OTHPER', 'RRab', 'Class']]

#	print('FINAL PREDICTIONS')
#	print(final_predictions_df)
#	final_predictions_df.to_csv('{}/src/models/{}_{}/unknown_predictions_{}.csv'.format(project_dir, model_name, training_set, model_number), index=False)
#	print('Written to file.')
	return

def main():
	#make_model_table(model_name, training_set, som_dimension=2)
	SOM_and_RF_delta()

if __name__ == "__main__":
	main()
