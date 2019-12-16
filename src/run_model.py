# =============================================================================
# =============================================================================
# model.py
# Created by Chance Haycock December 2019
#
# A script to contain all models built over the course of the project.
# We first begin with applying the SOM and RF methods to a train/test split of
# campaign 3 and 4 to quanitify the models accuracy.
#
# =============================================================================
# =============================================================================

from kepler_data_utilities import *
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.ensemble import RandomForestClassifier as RCF
from sklearn.metrics import accuracy_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# ===================================
#   MODEL 1 - RandomForestClassifier
# ===================================

def SOM_and_RF(probability_threshold=0.5):
	c3_df = pd.read_csv("{}/tables/campaign_3_master_table_with_som_data.csv".format(project_dir))
	c4_df = pd.read_csv("{}/tables/campaign_4_master_table_with_som_data.csv".format(project_dir))
	df = c3_df.append(c4_df, ignore_index=True)

	# Restrict set to classifications > 0.5
	df = df[df['probability'] > probability_threshold]

	# XXX - REMOVE empty entries. Maybe try filling these later.
	# 22,000 vs 15, 000
	df = df.dropna()
	df = df.drop("probability", axis=1)


	training_split = 0.8
	train, test = train_test_split(df, stratify=df['class'], random_state=7, train_size=training_split)

	features = train.columns[1:23]

	classifier = RCF(random_state=2, n_estimators=100, class_weight='balanced', max_features=None)
	classifier.fit(train[features], train['class'])
	test_predictions = classifier.predict(test[features])
	fimp = classifier.feature_importances_
	plt.bar(features, fimp)
	plt.xticks(rotation=90)
	plt.tight_layout()
	plt.show()
	score = accuracy_score(test['class'], test_predictions)
	print("Model Score STRICT, Exactly right!: ", score)
	#print(classifier.predict_proba(test[features]))
	CM = confusion_matrix(test['class'], test_predictions)

	print("classified Stars: ")
	print("DSCUT: ", len(test[test['class'] == 'DSCUT']))
	print("EA: ", len(test[test['class'] == 'EA']))
	print("EB: ", len(test[test['class'] == 'EB']))
	print("GDOR: ", len(test[test['class'] == 'GDOR']))
	print("Noise: ", len(test[test['class'] == 'Noise']))
	print("OTHPER: ", len(test[test['class'] == 'OTHPER']))
	print("RRab: ", len(test[test['class'] == 'RRab']))

	#Normalise
	sumarr = CM.astype(np.float).sum(axis=1)
	
	normCM = [CM[i] / sumarr[i] for i in range(len(sumarr))] 
	cmlabels = np.sort(classifier.classes_) 
	sbn.heatmap(normCM, annot=True, linewidths=.5, cmap="YlGnBu", xticklabels=cmlabels, yticklabels=cmlabels)
	plt.xlabel("Predicted classes")
	plt.ylabel("True classes")
	plt.tight_layout()
	plt.show()

	return score


# =========================================
#    MODEL 2 - ExtraTreesClassifier
# =========================================
def SOM_and_ETC():

	_print_model_type("SOM and ETC")

	# Set training split here
	training_split = 0.75

	# Import global training data. Contains roughly 70 of each class.
	training_file = "{}/training_sets/c34_clean_1_RF.csv".format(project_dir)
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
	_plot_feature_importances(classifier, features)

	#=================================
	#  PROBABILTIES CONFUSION MATRIX
	#=================================
	_plot_confusion_matrix_with_probabilties(classifier, test, features)

	#=================================
	# CLASSIFICATIONS CONFUSION MATRIX
	#=================================
	_plot_confusion_matrix_with_classifications(classifier, test, test_predictions)

	#=================================
	#      BOTH CONFUSION MATRIX
	#=================================
#	_plot_confusion_matrices(classifier, test, test_predictions, features)

	#=================================
	#       CLASSIFIER SUMMARY
	#=================================
	_print_summary(test, test_predictions, training_file, training_split, score)

	return

def _plot_confusion_matrix_with_probabilties(classifier, test, features):
	pred_probs = classifier.predict_proba(test[features])

	DSCUT_prob = np.zeros(7)
	EA_prob = np.zeros(7)
	EB_prob = np.zeros(7)
	GDOR_prob = np.zeros(7)
	Noise_prob = np.zeros(7)
	OTHPER_prob = np.zeros(7)
	RRab_prob = np.zeros(7)

	for i in range(len(pred_probs)):
		actual_class = test['class'].iloc[i]
		if actual_class == "DSCUT": 
			DSCUT_prob += pred_probs[i]
		elif actual_class == "EA": 
			EA_prob += pred_probs[i]
		elif actual_class == "EB": 
			EB_prob += pred_probs[i]
		elif actual_class == "GDOR": 
			GDOR_prob += pred_probs[i]
		elif actual_class == "Noise": 
			Noise_prob += pred_probs[i]
		elif actual_class == "OTHPER": 
			OTHPER_prob += pred_probs[i]
		elif actual_class == "RRab": 
			RRab_prob += pred_probs[i]

	DSCUT_prob = DSCUT_prob / np.sum(DSCUT_prob)
	EA_prob = EA_prob / np.sum(EA_prob)
	EB_prob = EB_prob / np.sum(EB_prob)
	GDOR_prob = GDOR_prob / np.sum(GDOR_prob)
	Noise_prob = Noise_prob / np.sum(Noise_prob)
	OTHPER_prob = OTHPER_prob / np.sum(OTHPER_prob)
	RRab_prob = RRab_prob / np.sum(RRab_prob)

	CM_probabilities = []
	CM_probabilities.append(DSCUT_prob)
	CM_probabilities.append(EA_prob)
	CM_probabilities.append(EB_prob)
	CM_probabilities.append(GDOR_prob)
	CM_probabilities.append(Noise_prob)
	CM_probabilities.append(OTHPER_prob)
	CM_probabilities.append(RRab_prob)
	cmlabels = np.sort(classifier.classes_) 
	sbn.heatmap(CM_probabilities, annot=True, linewidths=.75, cmap="YlGnBu",
	            xticklabels=cmlabels, yticklabels=cmlabels, fmt=".2f",
	            center=0.5, vmin=0, vmax=1)
#	plt.xlabel("Predicted Classes")
#	plt.ylabel("True Classes")
	plt.tight_layout()
	plt.savefig("plots/RF_with_probabilities.eps", format='eps')
	plt.close()
	return None

def _plot_confusion_matrix_with_classifications(classifier, test, test_predictions):
	CM = confusion_matrix(test['class'], test_predictions)
	cmlabels = np.sort(classifier.classes_) 

	sumarr = CM.astype(np.float).sum(axis=1)
	normCM = [CM[i] / sumarr[i] for i in range(len(sumarr))] 
	sbn.heatmap(normCM, annot=True, linewidths=.75, cmap="YlGnBu",
	            xticklabels=cmlabels, yticklabels=cmlabels, fmt=".2f",
	            center=0.5, vmin=0, vmax=1)
	plt.xlabel("Predicted Classes")
	plt.ylabel("True Classes")
	plt.tight_layout()
#	plt.show()
	plt.savefig("plots/RF_with_classifications.eps", format='eps')
#	plt.close()
	return None


def _plot_confusion_matrices(classifier, test, test_predictions, features):

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

	pred_probs = classifier.predict_proba(test[features])

	DSCUT_prob = np.zeros(7)
	EA_prob = np.zeros(7)
	EB_prob = np.zeros(7)
	GDOR_prob = np.zeros(7)
	Noise_prob = np.zeros(7)
	OTHPER_prob = np.zeros(7)
	RRab_prob = np.zeros(7)

	for i in range(len(pred_probs)):
		actual_class = test['class'].iloc[i]
		if actual_class == "DSCUT": 
			DSCUT_prob += pred_probs[i]
		elif actual_class == "EA": 
			EA_prob += pred_probs[i]
		elif actual_class == "EB": 
			EB_prob += pred_probs[i]
		elif actual_class == "GDOR": 
			GDOR_prob += pred_probs[i]
		elif actual_class == "Noise": 
			Noise_prob += pred_probs[i]
		elif actual_class == "OTHPER": 
			OTHPER_prob += pred_probs[i]
		elif actual_class == "RRab": 
			RRab_prob += pred_probs[i]

	DSCUT_prob = DSCUT_prob / np.sum(DSCUT_prob)
	EA_prob = EA_prob / np.sum(EA_prob)
	EB_prob = EB_prob / np.sum(EB_prob)
	GDOR_prob = GDOR_prob / np.sum(GDOR_prob)
	Noise_prob = Noise_prob / np.sum(Noise_prob)
	OTHPER_prob = OTHPER_prob / np.sum(OTHPER_prob)
	RRab_prob = RRab_prob / np.sum(RRab_prob)

	CM_probabilities = []
	CM_probabilities.append(DSCUT_prob)
	CM_probabilities.append(EA_prob)
	CM_probabilities.append(EB_prob)
	CM_probabilities.append(GDOR_prob)
	CM_probabilities.append(Noise_prob)
	CM_probabilities.append(OTHPER_prob)
	CM_probabilities.append(RRab_prob)
	cmlabels = np.sort(classifier.classes_) 
	sbn.heatmap(CM_probabilities, annot=True, linewidths=.75, cmap="YlGnBu",
	            xticklabels=cmlabels, yticklabels=cmlabels, fmt=".2f",
	            center=0.5, vmin=0, vmax=1, cbar=False, ax=ax1)
	ax1.set_xlabel("Predicted Classes")
	ax1.set_ylabel("True Classes")

	CM = confusion_matrix(test['class'], test_predictions)

	sumarr = CM.astype(np.float).sum(axis=1)
	normCM = [CM[i] / sumarr[i] for i in range(len(sumarr))] 
	ax2 = sbn.heatmap(normCM, annot=True, linewidths=.75, cmap="YlGnBu",
	            xticklabels=cmlabels, yticklabels=cmlabels, fmt=".2f",
	            center=0.5, vmin=0, vmax=1, ax=ax2)
	



	ax2.set_xlabel("Predicted Classes")
	ax2.set_ylabel("True Classes")
	plt.tight_layout()
	plt.show()

#	plt.savefig("plots/RF_with_probabilities.eps", format='eps')
#	plt.close()




def _print_summary(test, test_predictions, training_file, training_split, score):
	print(classification_report(test['class'], test_predictions))
	print("Using training_set           :\t", training_file )
	print("Using train/test split       :\t{}/{}".format(training_split*100, 100 * (1.0 - training_split)))
	print("Model Accuracy Score (STRICT):\t %.3f" % score)

def _print_model_type(model_name):
	print("\t=================================================================")
	print("\t                        {} Model                        ".format(model_name))
	print("\t=================================================================\n")
	return None

def _plot_feature_importances(classifier, features):

	# Get feature importances
	fimp = classifier.feature_importances_

	# Get errors of feature importance
	std = np.std([tree.feature_importances_ for tree in classifier.estimators_],
	             axis=0)
	print(std)

	plt.bar(features, fimp, yerr=std)
	plt.xticks(rotation=90)
	plt.tight_layout()
	plt.show()
	return None


# =============================================================================
#                                    MAIN
# =============================================================================

def main():
	print("Main Program Running...\n")
#	SOM_and_RF()
	SOM_and_ETC()
#	array = []
#	x = np.linspace(0.5, 0.99, 20)
#	for point in x:
#		SOM_and_ETC(point)
#	plt.scatter(x, array)
#	plt.show()

if __name__ == "__main__":
	main()
