from kepler_data_utilities import *

def plot(detrending='k2sc', training_file='c1-4_beta'):
	# Import Training Epics
	training_epics_df = pd.read_csv('{}/training_sets/{}/{}.csv'\
	       .format(project_dir, detrending, training_file), 'r', delimiter=',')

	# Get rid of noise and make array of useful epics
	training_epics = training_epics_df['epic_number'].to_numpy()

	# Import ALL known phasefolded bins from MASTER table
	bins_1 = pd.read_csv('{}/tables/{}/campaign_1_master_table.csv'.format(project_dir, detrending))
	bins_2 = pd.read_csv('{}/tables/{}/campaign_2_master_table.csv'.format(project_dir, detrending))
	bins_3 = pd.read_csv('{}/tables/{}/campaign_3_master_table.csv'.format(project_dir, detrending))
	bins_4 = pd.read_csv('{}/tables/{}/campaign_4_master_table.csv'.format(project_dir, detrending))

	# Merge and Reduce
	phasefold_df = bins_1.append(bins_2).append(bins_3).append(bins_4)
	train_df = phasefold_df[phasefold_df['epic_number'].isin(training_epics)]
	needed_columns = make_bin_columns(64)
	print(train_df)

	for epic in train_df['epic_number']:
		row = train_df[train_df['epic_number'] == epic]
#		print(row)
		array = row[[x for x in needed_columns]].to_numpy()[0]
		star_class = row.iloc[0]['class']
		range = np.max(array) - np.min(array)
		if range > 1.0:
			print("Epic {} - Class {}".format(epic, star_class))
			print("Range: {}".format(range))
			print(array)
			print()



def main():
	plot()

if __name__ == "__main__":
	main()
