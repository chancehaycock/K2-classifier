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
from sklearn.ensemble import GradientBoostingClassifier as GB
from sklearn.ensemble import AdaBoostClassifier as ADA
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


# Function to decide which regime to train the model with.
def choose_features(train_df, include_som=True, include_period=True,
                    include_colour=True, include_absmag=True,
                    include_lc_stats=True, include_bin_lc_stats=True, som_1_dim=False):
	features = train_df.drop('class', axis=1).drop('probability', axis=1).columns[1:]

	if not include_som:
		if som_1_dim:
			som_columns = ["som_index", "template_dist"]
		else:
			som_columns = ["RRab_dist", "EA_dist", "EB_dist", "GDOR_DSCUT_dist", "template_dist"]
		features = features[~features.isin(som_columns)]
	if not include_period:
		period_columns = ["Period_1", "Period_2", "amp_ratio_21"]
		features = features[~features.isin(period_columns)]
	if not include_colour:
		colour_columns = ["bp_rp", "bp_g", "g_rp"]
		features = features[~features.isin(colour_columns)]
	if not include_absmag:
		absmag_columns = ["abs_magnitude"]
		features = features[~features.isin(absmag_columns)]
	if not include_lc_stats:
		lc_stats_columns = ["lc_amplitude", "p2p_98", "p2p_mean", "stddev", "kurtosis", "skew", "iqr", "mad"]
		features = features[~features.isin(lc_stats_columns)]
	if not include_bin_lc_stats:
		bin_lc_stats_columns = ["max_binned_p2p", "mean_binned_p2p"]
		features = features[~features.isin(bin_lc_stats_columns)]

	return features

# Combines SOM data with master table to create a table ready for the RF
def make_model_table(model_name, training_label, som_dimension=2):
	# XXX IMPORTANT - Has to bring in the class form the training set as that's the only point in the pipeline where class
	# labels have been changed
	training_set = pd.read_csv('{}/training_sets/k2sc/c1-4_{}.csv'.format(project_dir, training_label))[['epic_number', 'Class']]
	training_set['Class'] = [epic.strip() for epic in training_set['Class']]
	training_epics = training_set['epic_number'].to_numpy()

	master_c1 = pd.read_csv('{}/tables/k2sc/campaign_1_master_table.csv'.format(project_dir))
	master_c2 = pd.read_csv('{}/tables/k2sc/campaign_2_master_table.csv'.format(project_dir))
	master_c3 = pd.read_csv('{}/tables/k2sc/campaign_3_master_table.csv'.format(project_dir))
	master_c4 = pd.read_csv('{}/tables/k2sc/campaign_4_master_table.csv'.format(project_dir))

	data_master = master_c1.append(master_c2, ignore_index=True).append(master_c3, ignore_index=True).append(master_c4, ignore_index=True)
	bin_columns = make_bin_columns(64)
	data_master.rename(columns = {'class':'DJA_Class'}, inplace = True)
	data_master = data_master.drop(bin_columns, axis=1)
	data_train = data_master[data_master['epic_number'].isin(training_epics)]

	som_c1 = pd.read_csv('{}/som_statistics/k2sc/c1-4_{}/campaign_1_{}D.csv'.format(project_dir, training_label, som_dimension))
	som_c2 = pd.read_csv('{}/som_statistics/k2sc/c1-4_{}/campaign_2_{}D.csv'.format(project_dir, training_label, som_dimension))
	som_c3 = pd.read_csv('{}/som_statistics/k2sc/c1-4_{}/campaign_3_{}D.csv'.format(project_dir, training_label, som_dimension))
	som_c4 = pd.read_csv('{}/som_statistics/k2sc/c1-4_{}/campaign_4_{}D.csv'.format(project_dir, training_label, som_dimension))
	som_master = som_c1.append(som_c2, ignore_index=True).append(som_c3, ignore_index=True).append(som_c4, ignore_index=True)
	som_train = som_master[som_master['epic_number'].isin(training_epics)]

	train_df = data_train.merge(som_train, how='left', on='epic_number')
	train_df = train_df.merge(training_set, how='left', on='epic_number')
	train_df.rename(columns = {'Class':'class'}, inplace = True)
	train_df.to_csv('{}/src/models/{}_{}/train.csv'.format(project_dir, model_name, training_label), index=False)
	print(train_df)
	print(train_df.dropna())
	print('Model Table Created!')
	return


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
def evaluate_model(model_name, training_set, model_number, blank_classifier, data,
                   parameters, num_parameters, features, in_cv=5, out_cv=5, umbrella_train_size=500):

	# Quick downsampling of larger sized classes
	print("Downsampling OTHPER and Noise to {} members.".format(umbrella_train_size))
	if len(data[data['class'] == 'OTHPER']) > umbrella_train_size:
		data_without_umb = data[(data['class'] != 'OTHPER') & (data['class'] != 'Noise')]
		othper_sample = data[data['class'] == 'OTHPER'].sample(n=umbrella_train_size, random_state=3)
		noise_sample = data[data['class'] == 'Noise'].sample(n=umbrella_train_size, random_state=3)
		data = data_without_umb.append(othper_sample, ignore_index=True).append(noise_sample, ignore_index=True)

	print(data)
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

	macro_auc_ovo_scores     = np.zeros([out_cv])
	weighted_auc_ovo_scores  = np.zeros([out_cv])
	macro_auc_ovr_scores     = np.zeros([out_cv])
	weighted_auc_ovr_scores  = np.zeros([out_cv])

#	brier_loss_scores = np.zeros([out_cv])

	macro_f1_scores = np.zeros([out_cv])
	weighted_f1_scores = np.zeros([out_cv])

	macro_fbeta_scores = np.zeros([out_cv])
	weighted_fbeta_scores = np.zeros([out_cv])

	log_loss_scores = np.zeros([out_cv])

	macro_precision_scores = np.zeros([out_cv])
	weighted_precision_scores = np.zeros([out_cv])

	macro_recall_scores = np.zeros([out_cv])
	weighted_recall_scores = np.zeros([out_cv])

	accuracy_scores = np.zeros([out_cv])

	normCM = np.zeros([out_cv, 7, 7]) 
	feature_importances = np.zeros([out_cv, num_features])
	parameter_scores = np.zeros([out_cv, num_parameters])

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
		grid_search = GridSearchCV(estimator=blank_classifier, param_grid=parameters, cv=inner_cv)
		grid_search.fit(train_features, train_classes)
		# Get best params
		best_params = grid_search.best_params_

		# Collect all parameter scores - average at the end
		parameter_scores[i] = grid_search.cv_results_['mean_test_score']

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

		macro_auc_ovo_scores[i]    = roc_auc_score(test_classes, test_prediction_probs, labels=class_labels, multi_class="ovo", average="macro")
		weighted_auc_ovo_scores[i] = roc_auc_score(test_classes, test_prediction_probs, labels=class_labels, multi_class="ovo", average="weighted")
		macro_auc_ovr_scores[i]    = roc_auc_score(test_classes, test_prediction_probs, labels=class_labels, multi_class="ovr", average="macro")
		weighted_auc_ovr_scores[i] = roc_auc_score(test_classes, test_prediction_probs, labels=class_labels, multi_class="ovr", average="weighted")
#		brier_loss_scores[i] = brier_score_loss(test_classes, test_predictions)
		macro_f1_scores[i]         = f1_score(test_classes, test_predictions, average='macro')
		weighted_f1_scores[i]      = f1_score(test_classes, test_predictions, average='weighted')
		macro_fbeta_scores[i]      = fbeta_score(test_classes, test_predictions, average='macro', beta=0.5)
		weighted_fbeta_scores[i]   = fbeta_score(test_classes, test_predictions, average='weighted', beta=0.5)
		log_loss_scores[i]         = log_loss(test_classes, test_prediction_probs, labels=class_labels)
		macro_precision_scores[i]  = precision_score(test_classes, test_predictions, average='macro')
		weighted_precision_scores[i] = precision_score(test_classes, test_predictions, average='weighted')
		macro_recall_scores[i]     = recall_score(test_classes, test_predictions, average='macro')
		weighted_recall_scores[i]   = recall_score(test_classes, test_predictions, average='weighted')

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

	macro_auc_ovo_score_avg = np.mean(macro_auc_ovo_scores)
	macro_auc_ovo_score_std = np.std(macro_auc_ovo_scores)
	weighted_auc_ovo_score_avg = np.mean(weighted_auc_ovo_scores)
	weighted_auc_ovo_score_std = np.std(weighted_auc_ovo_scores)
	macro_auc_ovr_score_avg = np.mean(macro_auc_ovr_scores)
	macro_auc_ovr_score_std = np.std(macro_auc_ovr_scores)
	weighted_auc_ovr_score_avg = np.mean(weighted_auc_ovr_scores)
	weighted_auc_ovr_score_std = np.std(weighted_auc_ovr_scores)

#	brier_score_avg = np.mean(brier_scores)
#	brier_score_std = np.std(brier_scores)

	macro_f1_score_avg = np.mean(macro_f1_scores)
	macro_f1_score_std = np.std(macro_f1_scores)
	weighted_f1_score_avg = np.mean(weighted_f1_scores)
	weighted_f1_score_std = np.std(weighted_f1_scores)

	macro_fbeta_score_avg = np.mean(macro_fbeta_scores)
	macro_fbeta_score_std = np.std(macro_fbeta_scores)
	weighted_fbeta_score_avg = np.mean(weighted_fbeta_scores)
	weighted_fbeta_score_std = np.std(weighted_fbeta_scores)

	log_loss_score_avg = np.mean(log_loss_scores)
	log_loss_score_std = np.std(log_loss_scores)

	macro_precision_score_avg = np.mean(macro_precision_scores)
	macro_precision_score_std = np.std(macro_precision_scores)
	weighted_precision_score_avg = np.mean(weighted_precision_scores)
	weighted_precision_score_std = np.std(weighted_precision_scores)

	macro_recall_score_avg = np.mean(macro_recall_scores)
	macro_recall_score_std = np.std(macro_recall_scores)
	weighted_recall_score_avg = np.mean(weighted_recall_scores)
	weighted_recall_score_std = np.std(weighted_recall_scores)

	CM_avg = np.mean(normCM, axis=0)
	CM_std = np.std(normCM, axis=0)
	fimp_avg = np.mean(feature_importances, axis=0)
	fimp_std = np.std(feature_importances, axis=0)

	parameter_scores_avg = np.mean(parameter_scores, axis=0)
	parameter_scores_std = np.std(parameter_scores, axis=0)

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
	plt.yticks(rotation=0)
	plt.xlabel("Predicted Classes")
	plt.ylabel("True Classes")
	plt.tight_layout()
	cm_file = '{}/src/models/{}_{}/confusion_matrix_{}.pdf'.format(project_dir, model_name, training_set, model_number)
	plt.savefig('{}'.format(cm_file), format='pdf')
	plt.close()
	print("\nConfusion Matrix plot saved at {}".format(cm_file))

	# ========================
	# Plot Feature Importances
	# ========================
	plt.bar(features, fimp_avg, yerr=fimp_std)
	plt.xticks(rotation=90)
	plt.tight_layout()
	fi_file = '{}/src/models/{}_{}/feature_importance_{}.pdf'.format(project_dir, model_name, training_set, model_number)
	plt.savefig('{}'.format(fi_file), format='pdf')
	plt.close()
	print("\nFeature Importance plot saved at {}".format(fi_file))


	# ========================
	#  Plot Parameter Scores 
	# ========================
	x = range(len(parameter_scores_avg))
	plt.bar(x, parameter_scores_avg, yerr=parameter_scores_std)
	plt.tight_layout()
	param_file = '{}/src/models/{}_{}/parameter_scores_{}.pdf'.format(project_dir, model_name, training_set, model_number)
	plt.savefig('{}'.format(param_file), format='pdf')
	plt.close()
	print("\nParameter plot saved at {}".format(param_file))

	# ==========================================
	#       Return Overall Model Scores 
	# ==========================================

	print("\n=================== Average Cross Validated Scores ==================\n")
	print('\n             Macro AUC Score (OVO): %.3f +/- %.4f' % (macro_auc_ovo_score_avg, macro_auc_ovo_score_std))
	print('\n          Weighted AUC Score (OVO): %.3f +/- %.4f' % (weighted_auc_ovo_score_avg, weighted_auc_ovo_score_std))
	print('\n             Macro AUC Score (OVR): %.3f +/- %.4f' % (macro_auc_ovr_score_avg, macro_auc_ovr_score_std))
	print('\n          Weighted AUC Score (OVR): %.3f +/- %.4f' % (weighted_auc_ovr_score_avg, weighted_auc_ovr_score_std))
#	print('                         Brier Score: %.3f +/- %.4f' % (brier_score_avg, brier_score_std))
	print('\n                    Macro F1 Score: %.3f +/- %.4f' % (macro_f1_score_avg, macro_f1_score_std))
	print('\n                 Weighted F1 Score: %.3f +/- %.4f' % (weighted_f1_score_avg, weighted_f1_score_std))
	print('\n                 Macro Fbeta Score: %.3f +/- %.4f' % (macro_fbeta_score_avg, macro_fbeta_score_std))
	print('\n              Weighted Fbeta Score: %.3f +/- %.4f' % (weighted_fbeta_score_avg, weighted_fbeta_score_std))
	print('\n                    Log Loss Score: %.3f +/- %.4f' % (log_loss_score_avg, log_loss_score_std))
	print('\n             Macro Precision Score: %.3f +/- %.4f' % (macro_precision_score_avg, macro_precision_score_std))
	print('\n          Weighted Precision Score: %.3f +/- %.4f' % (weighted_precision_score_avg, weighted_precision_score_std))
	print('\n                Macro Recall Score: %.3f +/- %.4f' % (macro_recall_score_avg, macro_recall_score_std))
	print('\n             Weighted Recall Score: %.3f +/- %.4f' % (weighted_recall_score_avg, weighted_recall_score_std))
	print('\n              Model Accuracy Score: %.3f +/- %.4f\n' % (accuracy_score_avg, accuracy_score_std))

	print("\n\n\nNow using cross validation to find optimal hyper-parameters for the whole dataset.")

	final_cv = StratifiedKFold(n_splits=out_cv)
	final_grid_search = GridSearchCV(estimator=blank_classifier, param_grid=parameters, cv=final_cv, n_jobs=2)
	final_grid_search.fit(X, y)
	# Get best params
	final_best_params = final_grid_search.best_params_

	print('\nBest parameters for whole data set: {}'.format(final_best_params))

	print("\n\nModel Evaluation Complete.")

	return


def evaluate_model_simpleCV(model_name, training_set, model_number, classifier, data,
                            features, cv=10, umbrella_train_size=500, num_classifiers=50):

	print('Shuffling Data...')
	data = data.sample(frac=1)

	print("\nUsing simple cross validation...")
	print("CV Loop (Model Performance Estimation) using {} folds and a total of {} classifiers.".format(cv, num_classifiers))

	#==============
	# Features Used
	#==============
	num_features = len(features)
	print("\nUsing {} features:\n".format(num_features))
	for feat in features.to_numpy():
		print("{}".format(feat))

	# =========================================
	# Declare arrays ready for score collection
	# =========================================
	macro_auc_ovo_scores     = np.zeros([num_classifiers, cv])
	weighted_auc_ovo_scores  = np.zeros([num_classifiers, cv])
	macro_auc_ovr_scores     = np.zeros([num_classifiers, cv])
	weighted_auc_ovr_scores  = np.zeros([num_classifiers, cv])

#	brier_loss_scores = np.zeros([cv])

	macro_f1_scores = np.zeros([num_classifiers, cv])
	weighted_f1_scores = np.zeros([num_classifiers, cv])

	macro_fbeta_scores = np.zeros([num_classifiers, cv])
	weighted_fbeta_scores = np.zeros([num_classifiers, cv])

	log_loss_scores = np.zeros([num_classifiers, cv])

	macro_precision_scores = np.zeros([num_classifiers, cv])
	weighted_precision_scores = np.zeros([num_classifiers, cv])

	macro_recall_scores = np.zeros([num_classifiers, cv])
	weighted_recall_scores = np.zeros([num_classifiers, cv])

	accuracy_scores = np.zeros([num_classifiers, cv])

	normCM = np.zeros([num_classifiers, cv, 7, 7]) # Number of classes = 7
	feature_importances = np.zeros([num_classifiers, cv, num_features])
	summed_CM = np.zeros([7, 7])

	# ==================
	#     MAIN LOOP
	# ==================
	for i in range(num_classifiers):
		print('\nTraining Classifier {}'.format(i+1))

		# Quick downsampling of larger sized classes
		print("Downsampling OTHPER and Noise to {} members using df.sample(random_state={}).".format(umbrella_train_size, i))
		if len(data[data['class'] == 'OTHPER']) > umbrella_train_size:
			data_without_umb = data[(data['class'] != 'OTHPER') & (data['class'] != 'Noise')]
			othper_sample = data[data['class'] == 'OTHPER'].sample(n=umbrella_train_size, random_state=i)
			noise_sample = data[data['class'] == 'Noise'].sample(n=umbrella_train_size, random_state=i)
			data = data_without_umb.append(othper_sample, ignore_index=True).append(noise_sample, ignore_index=True)
		
		np.set_printoptions(precision=3)
		#====================
		# Split data
		#====================
		X = data[features]
		y = data['class']
		cv_iter = StratifiedKFold(n_splits=cv, random_state=123)

		for icv, (train_index, test_index) in enumerate(cv_iter.split(X, y)):
			# Divide up dataset
			train_features = X.iloc[train_index] 
			train_classes  = y.iloc[train_index] 
			test_features = X.iloc[test_index] 
			test_classes  = y.iloc[test_index] 

			classifier.fit(train_features, train_classes)
			test_predictions = classifier.predict(test_features)
			test_prediction_probs = classifier.predict_proba(test_features)
			class_labels = classifier.classes_

			# Append Scores
			accuracy_scores[i][icv]   = accuracy_score(test_classes, test_predictions)

			macro_auc_ovo_scores[i][icv]    = roc_auc_score(test_classes, test_prediction_probs, labels=class_labels, multi_class="ovo", average="macro")
			weighted_auc_ovo_scores[i][icv] = roc_auc_score(test_classes, test_prediction_probs, labels=class_labels, multi_class="ovo", average="weighted")
			macro_auc_ovr_scores[i][icv]    = roc_auc_score(test_classes, test_prediction_probs, labels=class_labels, multi_class="ovr", average="macro")
			weighted_auc_ovr_scores[i][icv] = roc_auc_score(test_classes, test_prediction_probs, labels=class_labels, multi_class="ovr", average="weighted")
#			brier_loss_scores[i] = brier_score_loss(test_classes, test_predictions)
			macro_f1_scores[i][icv]         = f1_score(test_classes, test_predictions, average='macro')
			weighted_f1_scores[i][icv]      = f1_score(test_classes, test_predictions, average='weighted')
			macro_fbeta_scores[i][icv]      = fbeta_score(test_classes, test_predictions, average='macro', beta=0.5)
			weighted_fbeta_scores[i][icv]   = fbeta_score(test_classes, test_predictions, average='weighted', beta=0.5)
			log_loss_scores[i][icv]         = log_loss(test_classes, test_prediction_probs, labels=class_labels)
			macro_precision_scores[i][icv]  = precision_score(test_classes, test_predictions, average='macro')
			weighted_precision_scores[i][icv] = precision_score(test_classes, test_predictions, average='weighted')
			macro_recall_scores[i][icv]     = recall_score(test_classes, test_predictions, average='macro')
			weighted_recall_scores[i][icv]   = recall_score(test_classes, test_predictions, average='weighted')


			# Append Confusion Matrices
			CM = confusion_matrix(test_classes, test_predictions)
			cmlabels = np.sort(classifier.classes_) 

			sumarr = CM.astype(np.float).sum(axis=1)
	#		normCM[i][icv] = [CM[j] / sumarr[j] for j in range(len(sumarr))] 
			normCM[i] = CM

			summed_CM += CM


			# Append Feature Importances
			feature_importances[i][icv] = classifier.feature_importances_ 

			# Print Report For Each Model XXX - THINK TWICE ABOUT PRINTING!
#			print("Report:")
#			print(classification_report(test_classes, test_predictions))

	# ====================================
	# Take averages over useful quantities
	# ====================================
	accuracy_score_avg = np.mean(accuracy_scores)
	accuracy_score_std = np.std(accuracy_scores)

	macro_auc_ovo_score_avg = np.mean(macro_auc_ovo_scores)
	macro_auc_ovo_score_std = np.std(macro_auc_ovo_scores)
	weighted_auc_ovo_score_avg = np.mean(weighted_auc_ovo_scores)
	weighted_auc_ovo_score_std = np.std(weighted_auc_ovo_scores)
	macro_auc_ovr_score_avg = np.mean(macro_auc_ovr_scores)
	macro_auc_ovr_score_std = np.std(macro_auc_ovr_scores)
	weighted_auc_ovr_score_avg = np.mean(weighted_auc_ovr_scores)
	weighted_auc_ovr_score_std = np.std(weighted_auc_ovr_scores)

#	brier_score_avg = np.mean(brier_scores)
#	brier_score_std = np.std(brier_scores)

	macro_f1_score_avg = np.mean(macro_f1_scores)
	macro_f1_score_std = np.std(macro_f1_scores)
	weighted_f1_score_avg = np.mean(weighted_f1_scores)
	weighted_f1_score_std = np.std(weighted_f1_scores)

	macro_fbeta_score_avg = np.mean(macro_fbeta_scores)
	macro_fbeta_score_std = np.std(macro_fbeta_scores)
	weighted_fbeta_score_avg = np.mean(weighted_fbeta_scores)
	weighted_fbeta_score_std = np.std(weighted_fbeta_scores)

	log_loss_score_avg = np.mean(log_loss_scores)
	log_loss_score_std = np.std(log_loss_scores)

	macro_precision_score_avg = np.mean(macro_precision_scores)
	macro_precision_score_std = np.std(macro_precision_scores)
	weighted_precision_score_avg = np.mean(weighted_precision_scores)
	weighted_precision_score_std = np.std(weighted_precision_scores)

	macro_recall_score_avg = np.mean(macro_recall_scores)
	macro_recall_score_std = np.std(macro_recall_scores)
	weighted_recall_score_avg = np.mean(weighted_recall_scores)
	weighted_recall_score_std = np.std(weighted_recall_scores)

	CM_avg = np.mean(normCM, axis=(0,1))
	CM_std = np.std(normCM, axis=(0,1))
	fimp_avg = np.mean(feature_importances, axis=(0,1))
	fimp_std = np.std(feature_importances, axis=(0,1))

	print('\nAverage Confusion Matrix over %d runs each on %d sampled training sets:' % (cv, num_classifiers))
	print(CM_avg)
	print('\nStd Dev of Confusion Matrix over %d runs each on %d sampled training sets:' % (cv, num_classifiers))
	print(CM_std)
	print('\nAverage Feature Importances over %d runs each on %d sampled training sets:' % (cv, num_classifiers))
	print(fimp_avg)
	print('\nStd Dev of  Feature Importances over %d runs each on %d sampled training sets:' % (cv, num_classifiers))
	print(fimp_std)


	print("SUMMED CM")
	print(summed_CM)

	# ========================
	# Plot Confusion Matrix
	# ========================
	sbn.heatmap(CM_avg, annot=True, linewidths=.75, cmap="YlGnBu",
	            xticklabels=cmlabels, yticklabels=cmlabels, fmt=".4f",
	            center=0.5, vmin=0, vmax=1)
	plt.yticks(rotation=0)
	plt.xlabel("Predicted Classes")
	plt.ylabel("True Classes")
	plt.tight_layout()
	cm_file = '{}/src/models/{}_{}/confusion_matrix_{}.pdf'.format(project_dir, model_name, training_set, model_number)
	plt.savefig('{}'.format(cm_file), format='pdf')
	plt.close()
	print("\nConfusion Matrix plot saved at {}".format(cm_file))

	# ========================
	# Plot Feature Importances
	# ========================
	plt.bar(features, fimp_avg, yerr=fimp_std)
	plt.xticks(rotation=90)
	plt.tight_layout()
	fi_file = '{}/src/models/{}_{}/feature_importance_{}.pdf'.format(project_dir, model_name, training_set, model_number)
	plt.savefig('{}'.format(fi_file), format='pdf')
	plt.close()
	print("\nFeature Importance plot saved at {}".format(fi_file))

	# ==========================================
	#       Return Overall Model Scores 
	# ==========================================

	print("\n=================== Average Cross Validated Scores ==================\n")
	print('\n             Macro AUC Score (OVO): %.3f +/- %.4f' % (macro_auc_ovo_score_avg, macro_auc_ovo_score_std))
	print('\n          Weighted AUC Score (OVO): %.3f +/- %.4f' % (weighted_auc_ovo_score_avg, weighted_auc_ovo_score_std))
	print('\n             Macro AUC Score (OVR): %.3f +/- %.4f' % (macro_auc_ovr_score_avg, macro_auc_ovr_score_std))
	print('\n          Weighted AUC Score (OVR): %.3f +/- %.4f' % (weighted_auc_ovr_score_avg, weighted_auc_ovr_score_std))
#	print('                         Brier Score: %.3f +/- %.4f' % (brier_score_avg, brier_score_std))
	print('\n                    Macro F1 Score: %.3f +/- %.4f' % (macro_f1_score_avg, macro_f1_score_std))
	print('\n                 Weighted F1 Score: %.3f +/- %.4f' % (weighted_f1_score_avg, weighted_f1_score_std))
	print('\n                 Macro Fbeta Score: %.3f +/- %.4f' % (macro_fbeta_score_avg, macro_fbeta_score_std))
	print('\n              Weighted Fbeta Score: %.3f +/- %.4f' % (weighted_fbeta_score_avg, weighted_fbeta_score_std))
	print('\n                    Log Loss Score: %.3f +/- %.4f' % (log_loss_score_avg, log_loss_score_std))
	print('\n             Macro Precision Score: %.3f +/- %.4f' % (macro_precision_score_avg, macro_precision_score_std))
	print('\n          Weighted Precision Score: %.3f +/- %.4f' % (weighted_precision_score_avg, weighted_precision_score_std))
	print('\n                Macro Recall Score: %.3f +/- %.4f' % (macro_recall_score_avg, macro_recall_score_std))
	print('\n             Weighted Recall Score: %.3f +/- %.4f' % (weighted_recall_score_avg, weighted_recall_score_std))
	print('\n              Model Accuracy Score: %.3f +/- %.4f\n' % (accuracy_score_avg, accuracy_score_std))

	return 0

