# =============================================================================
# =============================================================================
# som_and_rf_alpha.py
# Created by Chance Haycock January 2020
#
# Similar to Interim model but with RF and SOM. We have expanded the data set
# by using training data from campaigns 1, 2, 3, 4
#
# =============================================================================
# =============================================================================
from models.model_utilities import *

# =========================================
#      SOM AND RF (K2SC_ALPHA) MODEL
# =========================================

# First need to collate master table data and SOM data using training set alpha
def make_model_table():
	training_epics = pd.read_csv('{}/training_sets/k2sc/c1-4_alpha.csv'.format(project_dir))['epic_number'].to_numpy()

	master_c1 = pd.read_csv('{}/tables/k2sc/campaign_1_master_table.csv'.format(project_dir))
	master_c2 = pd.read_csv('{}/tables/k2sc/campaign_2_master_table.csv'.format(project_dir))
	master_c3 = pd.read_csv('{}/tables/k2sc/campaign_3_master_table.csv'.format(project_dir))
	master_c4 = pd.read_csv('{}/tables/k2sc/campaign_4_master_table.csv'.format(project_dir))
	data_master = master_c1.append(master_c2, ignore_index=True).append(master_c3, ignore_index=True).append(master_c4, ignore_index=True)
	bin_columns = make_bin_columns(64)
	data_master = data_master.drop(bin_columns, axis=1)
	data_train = data_master[data_master['epic_number'].isin(training_epics)]

	som_c1 = pd.read_csv('{}/som_statistics/k2sc/c1-4_alpha/campaign_1.csv'.format(project_dir))
	som_c2 = pd.read_csv('{}/som_statistics/k2sc/c1-4_alpha/campaign_2.csv'.format(project_dir))
	som_c3 = pd.read_csv('{}/som_statistics/k2sc/c1-4_alpha/campaign_3.csv'.format(project_dir))
	som_c4 = pd.read_csv('{}/som_statistics/k2sc/c1-4_alpha/campaign_4.csv'.format(project_dir))
	som_master = som_c1.append(som_c2, ignore_index=True).append(som_c3, ignore_index=True).append(som_c4, ignore_index=True)
	som_train = som_master[som_master['epic_number'].isin(training_epics)]

	train_df = data_train.merge(som_train, how='left', on='epic_number')
	train_df.to_csv('{}/src/models/som_and_rf_alpha/train.csv'.format(project_dir), index=False)
	print('Model Table Created!')
	print(len(train_df.columns))
	return


# Things that I want from an overall model
# - Overall score (means and variance for stability)
# - Confusion Matrix (means and variance)
# - Feature Importance
# - Learning Curve
# - summary f_1 scores?

def SOM_and_RF_alpha():

	model_number = sys.argv[1]

	print_model_type("SOM and Random Forest")

	# Import global training data. Contains roughly 100 of each class.
	training_file = "{}/src/models/som_and_rf_alpha/train.csv".format(project_dir)
	df = pd.read_csv(training_file)
	print("Using training file: {}".format(training_file))

	# Fill empty entries. Maybe try filling these later.
	df = df.fillna(-1)

	# Features to be tested. Column 0 is epics.
	features = df.drop('class', axis=1).drop('probability', axis=1).columns[1:len(df.columns)-2]

	blank_classifier = RCF(random_state=2, class_weight='balanced')
#	parameters = {'n_estimators':[300, 400, 500, 600],\
#	         'min_samples_split':[2, 3, 4, 5, 6],\
#	              'max_features':[4, 5, 6, 7, 8] }
	parameters = {'n_estimators':[300],\
	         'min_samples_split':[3],\
	              'max_features':[4] }
	evaluate_model(model_number, blank_classifier, df, parameters, features, in_cv=5, out_cv=5)

	# Do learning curve analysis here

	return


# =============================================================================
#                                    MAIN
# =============================================================================

def main():
#	make_model_table()
	SOM_and_RF_alpha()

if __name__ == "__main__":
	main()
