from kepler_data_utilities import *
from models.model_utilities import *

# Main function to run tained model on unseen data.
# i.e. K2SC campaigns 5, 6, 7, 8, 10, with GAIA data.

def make_test_table(model_name, training_set, som_dimension=2):
	train_df = pd.read_csv("{}/src/models/{}_{}/train.csv".format(project_dir, model_name, training_set))
	train_df = train_df.drop('class', axis=1).drop('DJA_Class', axis=1).drop('probability', axis=1)
	features = train_df.columns

	master_c5 = pd.read_csv(  '{}/tables/k2sc/campaign_5_master_table.csv'.format(project_dir))
	master_c5['Campaign'] = 5
	master_c6 = pd.read_csv(  '{}/tables/k2sc/campaign_6_master_table.csv'.format(project_dir))
	master_c6['Campaign'] = 6
	master_c7 = pd.read_csv(  '{}/tables/k2sc/campaign_7_master_table.csv'.format(project_dir))
	master_c7['Campaign'] = 7
	master_c8 = pd.read_csv(  '{}/tables/k2sc/campaign_8_master_table.csv'.format(project_dir))
	master_c8['Campaign'] = 8
	master_c10 = pd.read_csv('{}/tables/k2sc/campaign_10_master_table.csv'.format(project_dir))
	master_c10['Campaign'] = 10

	data_master = master_c5.append(master_c6,  ignore_index=True)\
	                       .append(master_c7,  ignore_index=True)\
	                       .append(master_c8,  ignore_index=True)\
	                       .append(master_c10, ignore_index=True)

	som_c5 = pd.read_csv(  '{}/som_statistics/k2sc/c1-4_{}/campaign_5_{}D.csv'.format(project_dir, training_set, som_dimension))
	som_c6 = pd.read_csv(  '{}/som_statistics/k2sc/c1-4_{}/campaign_6_{}D.csv'.format(project_dir, training_set, som_dimension))
	som_c7 = pd.read_csv(  '{}/som_statistics/k2sc/c1-4_{}/campaign_7_{}D.csv'.format(project_dir, training_set, som_dimension))
	som_c8 = pd.read_csv(  '{}/som_statistics/k2sc/c1-4_{}/campaign_8_{}D.csv'.format(project_dir, training_set, som_dimension))
	som_c10 = pd.read_csv('{}/som_statistics/k2sc/c1-4_{}/campaign_10_{}D.csv'.format(project_dir, training_set, som_dimension))

	som_test = som_c5.append(som_c6,  ignore_index=True)\
	                 .append(som_c7,  ignore_index=True)\
	                 .append(som_c8,  ignore_index=True)\
	                 .append(som_c10, ignore_index=True)

	test_df = data_master.merge(som_test, how='left', on='epic_number')
	print(test_df)
	features = features
	
	test_df = test_df[features]
	test_df['Campaign'] = data_master['Campaign']
	print(test_df)
	# Remove all entries with missing values
	test_df = test_df.dropna()
	print(test_df)

	test_df.to_csv('{}/src/models/{}_{}/test_with_camp.csv'.format(project_dir, model_name, training_set), index=False)
	print('Test Table Created!')
	return

def run_on_unseen(trained_classifier, model_name, training_set, test_df, features):
	test_features = test_df[features]
	test_prediction_probs = trained_classifier.predict_proba(test_features)
	return test_prediction_probs


def main():
	model_name = 'som_and_rf'
	training_set = 'delta'

	make_test_table(model_name, training_set)



if __name__ == "__main__":
	main()
