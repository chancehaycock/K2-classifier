# Opens files from the sources, period, non-period, gaia, known
# and exports into a campaign_{campaign_num}_master_table_{details}.csv

from kepler_data_utilities import *

# The aim of this script is to to produce a CSV file per Campaign including most
# of the relevant data to be fed into the machine learning model. 
#
# The table will consist of statistics for each lightcurve present in a
# campaign. Some of these will be directly calculated from the lightcurve
# (periods, ratios, etc.) and other quantities will be crossed matched from
# GAIA data as produced in gaia_crossmatch.py. Not all lightcurves present in
# the K2 data have corresponding cross-matches in GAIA, but most do.
# TODO - We will need to come up with some sort of flag within the table to
# show this. Models can then be run on any combination of LC-periodic,
# LC-Nonperiodic and GAIA data.  

# Everything stems from the period file.

# ========================================================
#              Dependencies (CSV Files required)
# ========================================================
	# - 1) Periods file as from SJH
	# - 2) Unique GAIA data - from gaia_crossmatch.py
	# - 3) Lightcurve statistics - from lightcurve_statistics.py
	# - 4) Phasefolded Bin values - from phasefold.py
	# - 5) Classes and Probabilites - from armstong_0_to_4.csv

# Array of gaia features to be used (Example for now)
gaia_features = ['epic_number', 'abs_magnitude', 'bp_rp', 'bp_g', 'g_rp'] 


def create_table(campaign_num, detrending):

	known_campaign = campaign_is_known(campaign_num)

	print("Loading Files...")

	# ============================
	# Load Data from the 4 sources
	# ============================

	# 1) Period Data
	period_file = '{}/periods/{}/campaign_{}.csv'.format(project_dir, detrending, campaign_num)
	period_df = pd.read_csv(period_file, low_memory=False)
	period_df = period_df[['epic_number', 'Period_1', 'Period_2']]
	# Add columns for ratios.
	period_df['amp_ratio_21'] = period_df['Period_2'] / period_df['Period_1']

	# 2) Gaia Data
	gaia_file = '{}/cross_match_data/gaia/unique_gaia_campaign_{}_data.csv'\
	             .format(project_dir, campaign_num)
	gaia_df = pd.read_csv(gaia_file, low_memory=False)
	# Calculate abs_magnitude here. Calculated from gaia_magnitude.
	gaia_df['abs_magnitude'] = 5.0 + gaia_df['phot_g_mean_mag']\
	                         - 5.0 * np.log10(gaia_df['r_est']) 

	# 3) Lightcurve Statistics
	lc_stats_file = '{}/lightcurve_statistics/{}/campaign_{}.csv'\
	                 .format(project_dir, detrending, campaign_num)
	lc_stats_df = pd.read_csv(lc_stats_file, low_memory=False)

	# 4) Phasefold Values
	phasefold_file = '{}/phasefold_bins/{}/campaign_{}_interpolated.csv'\
	                 .format(project_dir, detrending, campaign_num)
	phasefold_df = pd.read_csv(phasefold_file, low_memory=False)

	# 5) Classes and Probabilities
	if known_campaign:
		classes_file = '{}/known/armstrong_0_to_4.csv'.format(project_dir, campaign_num)
		classes_df = pd.read_csv(classes_file, low_memory=False) 

	print("Values calculated. (Absolute magnitude and amplitude ratios)")
	print("{} loaded.".format(period_file))
	print("{} loaded.".format(gaia_file))
	print("{} loaded.".format(lc_stats_file))
	print("{} loaded.".format(phasefold_file))
	print("{} loaded.".format(classes_file)) if known_campaign else None

	add_columns = True
	for i, epic in enumerate(period_df['epic_number']):
		# XXX issue epic
		if epic == 206413104:
			continue

		# Reduce dataframes to rows of only the interested epic. 
		# All should be non-empty apart from 4) which has delted entries with
		# periods of more than or equal to 20.
		# ===================================================================
		# 1) Periodic Features - SJH
		sub_period_df = period_df[period_df['epic_number'] == epic]
		# 2) GAIA Crossmatch = CJH
		sub_gaia_df = gaia_df[gaia_df['epic_number'] == epic][gaia_features]
		# 3) Lightcurve Statistics - CJH
		sub_lc_stats_df = lc_stats_df[lc_stats_df['epic_number'] == epic]
		# 4) Phasefold Bins - CJH
		sub_phasefold_df = phasefold_df[phasefold_df['epic_number'] == epic]
		# 5) Classes File (Only happens on known campaigns)
		if known_campaign:
			sub_classes_df = classes_df[classes_df['epic_number'] == epic]
			if (not sub_classes_df.empty):
				star_class = sub_classes_df.iloc[0]["Class"]
				probability = sub_classes_df.iloc[0][star_class]
				star_class = star_class.split()[0]
			else:
				# Some mismatch in data set Sizes (NEPTUNE) etc..
				star_class = "NOT CLASSIFIED"
				probability = -1

			class_columns = ["epic_number", "class", "probability"]
			overall_class_df = pd.DataFrame(columns=class_columns)

			series = pd.Series([epic, star_class, probability], index=class_columns)
			overall_class_df = overall_class_df.append(series, ignore_index=True)
		# ===================================================================

		# ===============================
		#      Merge All 5 Sources!
		# ===============================
		df = sub_period_df.merge(sub_gaia_df, how='left', on='epic_number')\
		              .merge(sub_lc_stats_df, how='left', on='epic_number')\
		              .merge(sub_phasefold_df, how='left', on='epic_number')

		if known_campaign:
			df = df.merge(overall_class_df,   how='left', on='epic_number')

		# ===============================
		#        WRITE IT TO CSV
		# ===============================
		with open('{}/tables/{}/campaign_{}_master_table.csv'\
		         .format(project_dir, detrending, campaign_num), 'a+') as file:
			if (add_columns):
				df.to_csv(file, index=None)
				add_columns = False
			else:
				df.to_csv(file, header=False, index=None)

		# Progress Bar
		if (i % 50 == 0):
			size = len(period_df["epic_number"])
			print("{0:.2f}%".format(i * 100 / size))

	print("Master table for campaign {} created.".format(campaign_num))
	print("Table has {} entries.".format(size))

def main():
	# Creates table of campaign(x) EPICS with columns:
	# period, etc...
#	create_table(1, 'k2sc')
#	create_table(2, 'k2sc')
#	create_table(3, 'k2sc')
#	create_table(4, 'k2sc')
	create_table(10, 'k2sc')
	create_table(6, 'k2sc')
	create_table(7, 'k2sc')
	create_table(8, 'k2sc')
	print('Program complete.')

if __name__ == "__main__":
	main()
