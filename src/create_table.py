from kepler_data_utilities import *

# The aim of this script is to to produce a CSV file per Campaign including all
# relevant data to be fed into the machine learning model. A full list of the
# columns will be present here:
#
#                      <INSERT COLUMNS OF TABLE>
#
# The table will consist of statistics for each lightcurve present in a
# campaign. Some of these will be directly calculated from the lightcurve
# (periods, ratios, etc.) and other quantities will be crossed matched from
# GAIA data as produced in gaia_crossmatch.py. Not all lightcurves present in
# the K2 data have corresponding cross-matches in GAIA, but most do.
# TODO - We will need to come up with some sort of flag within the table to
# show this. Models can then be run on any combination of LC-periodic,
# LC-Nonperiodic and GAIA data.  


# K2ID, Period1..6, Ratios, temp, r_est, abs_magnitude etc.

# Array of features to be used (Example for now)
gaia_features = ['epic_number',
                 'r_est'  , 'r_lo'       , 'r_hi',
                 'k2_teff', 'k2_tefferr1', 'k2_tefferr2',
                 'k2_rad' , 'k2_raderr1' , 'k2_raderr2' ,
                 'k2_mass', 'k2_masserr1', 'k2_masserr2',
                 'k2_kepmag'] 

project_dir = "/Users/chancehaycock/dev/machine_learning/px402"

def create_table(campaign_num):
	if campaign_num in [0, 1, 2, 3, 4]:
		known_campaign = True
	else:
		known_campaign = False

	print("Loading Files...")
	# Load Kepler ID's
	kid_file = '{}/kepler_ids/k2sc_c{}_id.csv'.format(project_dir,
	                                                  campaign_num)
	kep_ids = open(kid_file, 'r')
	# Load Data from the 3/4 sources
	period_file = '{}/cross_match_data/gaia/sample_sjh_{}.csv'\
	              .format(project_dir, campaign_num)
	period_df = pd.read_csv(period_file)

	gaia_file = '{}/cross_match_data/gaia/unique_gaia_campaign_{}_data.csv'\
	             .format(project_dir, campaign_num)
	gaia_df = pd.read_csv(gaia_file, low_memory=False)
	# non_period_file = ...

	# Classes
	if known_campaign:
		classes_file = '{}/known/armstrong_0_to_4.csv'.format(project_dir)
		classes_df = pd.read_csv(classes_file)
	print("Files Loaded.")

	add_columns = True
	for epic in period_df['epic_number']:
		# 1) Periodic Features - SJH
		sub_period_df = period_df[period_df['epic_number'] == epic]
		# 2) Non-Perioid Features - CJH
		# TODO
		# 3) GAIA Crossmatch = CJH
		sub_gaia_df = gaia_df[gaia_df['epic_number'] == epic][gaia_features]
		# 4) Known Classes Crossmatch
		if known_campaign:
			sub_classes_df = classes_df[classes_df['epic_number'] == epic]\
		                                         [['epic_number', 'Class']]
		# Merge them!
		total_df = sub_period_df.merge(sub_gaia_df, how='left',
		                               on='epic_number')
		if known_campaign:
			total_df = total_df.merge(sub_classes_df, how='left',
			                          on='epic_number')
		with open('{}/cross_match_data/gaia/campaign_{}_mixed_table.csv'\
		         .format(project_dir, campaign_num), 'a+') as file:
			if (add_columns):
				total_df.to_csv(file, index=None)
				add_columns = False
			else:
				total_df.to_csv(file, header=False, index=None)
		print("Added EPIC {} to table.".format(epic))
	kep_ids.close()
	print("Table created.")

def main():
	# Creates table of campaign(x) EPICS with columns:
	# period, etc...
#	create_table(3)
#	create_table(4)
	create_table(5)
	print('Program complete.')

if __name__ == "__main__":
	main()
