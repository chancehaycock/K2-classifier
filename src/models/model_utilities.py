# =============================================================================
# =============================================================================
# model_utilities.py
# Created by Chance Haycock December 2019
#
# All functionality required by other models
# =============================================================================
# =============================================================================

from kepler_data_utilities import *
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.ensemble import RandomForestClassifier as RCF
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold


def plot_confusion_matrix_with_probabilties(classifier, test, features):
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
	plt.xlabel("Predicted Classes")
	plt.ylabel("True Classes")
	plt.tight_layout()
	plt.show()
#	plt.savefig("plots/RF_with_probabilities.eps", format='eps')
#	plt.close()
	return None

def plot_confusion_matrix_with_classifications(classifier, test, test_predictions):
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
	plt.show()
#	plt.savefig("plots/RF_with_classifications.eps", format='eps')
#	plt.close()
	return None

def print_summary(test, test_predictions, training_file, training_split, score):
	print(classification_report(test['class'], test_predictions))
	print("Using training_set           :\t", training_file )
	print("Using train/test split       :\t{}/{}".format(training_split*100, 100 * (1.0 - training_split)))
	print("Model Accuracy Score (STRICT):\t %.3f" % score)

def print_model_type(model_name):
	print("================================================================================")
	print("                               {} Model                        ".format(model_name))
	print("================================================================================\n")
	return None

def plot_feature_importances(classifier, features):

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


#=================================
#    Learning Curve Function 
#=================================
# Works on different sizes training data. 
# Splits the training set into cv folds. Performs cross validation.
# train_scores is the average scores of the model when performed on the set that was used
# to train it - obviousy does well
# test_scores is the average score of the remaining sets.
def plot_learning_curve(classifier, data, features, cv=5):
	train_sizes = np.linspace(0.1, 1.0, 25)
	train_sizes, train_scores, test_scores = \
	     learning_curve(classifier, data[features], data['class'], cv=cv, train_sizes=train_sizes)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.05, color='red')
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.051, color='green')
	plt.plot(train_sizes, train_scores_mean,'o-', color='red', ms=2.5, linewidth=1.0, label='Training Set')

	plt.plot(train_sizes, test_scores_mean, 'o-', color='green', ms=2.5, linewidth=1.0, label='Cross Validation Set')
	plt.xlabel("Training Set Size")
	plt.ylabel("Score")
	plt.legend()
	plt.show()
	return

# ===============================
# Main Model Evaluation Function
# ==============================
# Use this to tune hyper-parameters and then get model score with nested validation
def evaluate_model(model_number, blank_classifier, data, parameters, features, in_cv=5, out_cv=5):

	np.set_printoptions(precision=3)
	#=================================
	# Split data into inner, outer cv
	#=================================
	X = data[features]
	y = data['class']
	inner_rand_state = 123
	outer_rand_state = 123
	inner_cv = StratifiedKFold(n_splits=in_cv, random_state=inner_rand_state)
	outer_cv = StratifiedKFold(n_splits=out_cv, random_state=outer_rand_state)
	print("\nUsing nested cross validation...")
	print("Outer Loop (Model Performance Estimation) using {} folds. Random State: {}".format(out_cv, outer_rand_state))
	print("Inner Loop (Hyper-parameter optimisation) using {} folds. Random State: {}".format(in_cv, inner_rand_state))

	#==============
	# Features Used
	#==============
	num_features = len(features)
	print("\nUsing {} features:\n".format(num_features))
	for feat in features.to_numpy():
		print("{}".format(feat))

	accuracy_scores = np.zeros([out_cv])
#	auc_scores = np.zeros([out_cv])
#	brier_loss_scores = np.zeros([out_cv])
#	f1_scores = np.zeros([out_cv])
#	f_beta_scores = np.zeros([out_cv])
#	log_loss_scores = np.zeros([out_cv])
#	precision_scores = np.zeros([out_cv])
#	recall_scores = np.zeros([out_cv])

	normCM = np.zeros([out_cv, 7, 7]) 
	feature_importances = np.zeros([out_cv, num_features])

	# ===============
	# Parameter Space
	# ===============
	print("\nInner loop is optimising over parameter space:")
	print("{}\n\n".format(parameters))

	for i, (outer_train_index, outer_test_index) in enumerate(outer_cv.split(X, y)):
		# Divide up dataset
		train_features = X.iloc[outer_train_index] 
		train_classes  = y.iloc[outer_train_index] 
		test_features = X.iloc[outer_test_index] 
		test_classes  = y.iloc[outer_test_index] 
		# Grid search over inner train set to get best params
		grid_search = GridSearchCV(estimator=blank_classifier, param_grid=parameters, cv=inner_cv, n_jobs=2)
		grid_search.fit(train_features, train_classes)
		# Get best params
		best_params = grid_search.best_params_
		print('Best Parameters for outer loop {}: {}\n'.format(i, best_params))
		max_features = best_params['max_features']
		min_samples_split = best_params['min_samples_split']
		n_estimators = best_params['n_estimators']

		# Use these best params to test the model. Can continue here as normal
		best_classifier = RCF(n_estimators=n_estimators, min_samples_split=min_samples_split,
		                      max_features=max_features, class_weight='balanced', random_state=123)

		best_classifier.fit(train_features, train_classes)
		test_predictions = best_classifier.predict(test_features)
		test_prediction_probs = best_classifier.predict_proba(test_features)
		class_labels = best_classifier.classes_

		# Append Scores
		accuracy_scores[i]   = accuracy_score(test_classes, test_predictions)
		auc_scores[i]        = roc_auc_score(test_classes, test_prediction_probs) 
#		brier_loss_scores[i] = brier_score_loss(test_classes, test_predictions)
#		f1_scores[i]         = f1_score(test_classes, test_predictions, average=None)
#		f_beta_scores[i]     = fbeta_score(test_classes, test_predictions, average=None)
#		log_loss_scores[i]   = log_loss(test_classes, test_predictions)
#		precision_scores[i]  = precision_score(test_classes, test_predictions)
#		recall_scores[i]     = recall_score(test_classes, test_predictions)

		# Append Confusion Matrices
		CM = confusion_matrix(test_classes, test_predictions)
		cmlabels = np.sort(best_classifier.classes_) 

		sumarr = CM.astype(np.float).sum(axis=1)
		normCM[i] = [CM[j] / sumarr[j] for j in range(len(sumarr))] 

		# Append Feature Importances
		feature_importances[i] = best_classifier.feature_importances_ 

		# Print Report For Each Model
		print("Report:")
		print(classification_report(test_classes, test_predictions))

	# ====================================
	# Take averages over useful quantities
	# ====================================
	accuracy_score_avg = np.mean(accuracy_scores)
	accuracy_score_std = np.std(accuracy_scores)
	auc_score_avg = np.mean(auc_scores)
	auc_score_std = np.std(auc_scores)
#	brier_score_avg = np.mean(brier_scores)
#	brier_score_std = np.std(brier_scores)
#	f1_score_avg = np.mean(f1_scores)
#	f1_score_std = np.std(f1_scores)
#	f_beta_score_avg = np.mean(f_beta_scores)
#	f_beta_score_std = np.std(f_beta_scores)
#	log_loss_score_avg = np.mean(log_loss_scores)
#	log_loss_score_std = np.std(log_loss_scores)
#	precision_score_avg = np.mean(precision_scores)
#	precision_score_std = np.std(precision_scores)
#	recall_score_avg = np.mean(recall_scores)
#	recall_score_std = np.std(recall_scores)

	CM_avg = np.mean(normCM, axis=0)
	CM_std = np.std(normCM, axis=0)
	fimp_avg = np.mean(feature_importances, axis=0)
	fimp_std = np.std(feature_importances, axis=0)

	print('\nAverage Confusion Matrix over %d runs:' % out_cv)
	print(CM_avg)
	print('\nStd Dev of Confusion Matrix over %d runs:' % out_cv)
	print(CM_std)
	print('\nAverage Feature Importances over %d runs:' % out_cv)
	print(fimp_avg)
	print('\nStd Dev of  Feature Importances over %d runs:' % out_cv)
	print(fimp_std)

	# ========================
	# Plot Confusion Matrix
	# ========================
	sbn.heatmap(CM_avg, annot=True, linewidths=.75, cmap="YlGnBu",
	            xticklabels=cmlabels, yticklabels=cmlabels, fmt=".4f",
	            center=0.5, vmin=0, vmax=1)
	plt.xlabel("Predicted Classes")
	plt.ylabel("True Classes")
	plt.tight_layout()
	cm_file = '{}/src/models/som_and_rf_alpha/confusion_matrix_{}.pdf'.format(project_dir, model_number)
	plt.savefig('{}'.format(cm_file), format='pdf')
	plt.close()
	print("\nConfusion Matrix plot saved at {}".format(cm_file))

	# ========================
	# Plot Feature Importances
	# ========================
	plt.bar(features, fimp_avg, yerr=fimp_std)
	plt.xticks(rotation=90)
	plt.tight_layout()
	fi_file = '{}/src/models/som_and_rf_alpha/feature_importance_{}.pdf'.format(project_dir, model_number)
	plt.savefig('{}'.format(fi_file), format='pdf')
	plt.close()
	print("\nFeature Importance plot saved at {}".format(fi_file))

	# ==========================================
	#  Return Overall Model Generalisation Score 
	# ==========================================
	print('\n                         AUC Score: %.3f +/- %.4f' % (auc_score_avg, auc_score_std))
#	print('                         Brier Score: %.3f +/- %.4f' % (brier_score_avg, brier_score_std))
#	print('                            F1 Score: %.3f +/- %.4f' % (f1_score_avg, f1_score_std))
#	print('                        F Beta Score: %.3f +/- %.4f' % (f_beta_score_avg, f_beta_score_std))
#	print('                      Log Loss Score: %.3f +/- %.4f' % (log_loss_score_avg, log_loss_score_std))
#	print('                     Precision Score: %.3f +/- %.4f' % (precision_score_avg, precision_score_std))
#	print('                        Recall Score: %.3f +/- %.4f' % (recall_score_avg, recall_score_std))
	print('\nOverall Model Generalisation Score:\n%.3f +/- %.4f\n' % (accuracy_score_avg, accuracy_score_std))

	print("\n\n\nNow using cross validation to find optimal hyper-parameters for the whole dataset.")

	final_cv = StratifiedKFold(n_splits=out_cv, random_state=123)
	final_grid_search = GridSearchCV(estimator=blank_classifier, param_grid=parameters, cv=final_cv, n_jobs=2)
	final_grid_search.fit(X, y)
	# Get best params
	final_best_params = final_grid_search.best_params_

	print('\nBest parameters for whole data set: {}'.format(final_best_params))

	print("\n\nModel Evaluation Complete.")

	return

