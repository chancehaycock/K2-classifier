# The aim of this script is to take in master_tables 3 and 4 and restrict them
# to a training set ready for use in all models.o

from kepler_data_utilities import *

def make_training_set(detrending, training_label, write_to_csv=False):

	avail_epics = pd.read_csv("{}/training_sets/{}/everything.csv".format(project_dir, detrending))["epic_number"].to_numpy()

	# Open Known Files
	data_file = "{}/known/armstrong_0_to_4.csv".format(project_dir)

	# Import GAIA data so that we can make a training set which has all features availalable - r_est, colour.
	gaia_file_1 = "{}/cross_match_data/gaia/unique_gaia_campaign_1_data.csv".format(project_dir)
	gaia_file_2 = "{}/cross_match_data/gaia/unique_gaia_campaign_2_data.csv".format(project_dir)
	gaia_file_3 = "{}/cross_match_data/gaia/unique_gaia_campaign_3_data.csv".format(project_dir)
	gaia_file_4 = "{}/cross_match_data/gaia/unique_gaia_campaign_4_data.csv".format(project_dir)
	gaia_df_1 = pd.read_csv(gaia_file_1, low_memory=False)
	gaia_df_2 = pd.read_csv(gaia_file_2, low_memory=False)
	gaia_df_3 = pd.read_csv(gaia_file_3, low_memory=False)
	gaia_df_4 = pd.read_csv(gaia_file_4, low_memory=False)
	gaia_df = gaia_df_1.append(gaia_df_2).append(gaia_df_3).append(gaia_df_4)
	gaia_df = gaia_df[['epic_number', 'r_est', 'bp_rp', 'bp_g', 'g_rp']]
	gaia_df = gaia_df.dropna()
	gaia_epics = gaia_df['epic_number'].to_numpy()

	df   = pd.read_csv(data_file)
	df = df[df["Campaign"] != 0]
	df = df[df['epic_number'].isin(avail_epics)]

	df['Probability'] = df[['  RRab', '    EA', '    EB', ' DSCUT', '  GDOR', 'OTHPER', ' Noise']].max(axis=1)

	RRab_df   = df[df["Class"] == "  RRab"].dropna().sort_values("  RRab", ascending=False)
	EA_df     = df[df["Class"] == "    EA"].dropna().sort_values("    EA", ascending=False)
	EB_df     = df[df["Class"] == "    EB"].dropna().sort_values("    EB", ascending=False)
	DSCUT_df  = df[df["Class"] == " DSCUT"].dropna().sort_values(" DSCUT", ascending=False)
	GDOR_df   = df[df["Class"] == "  GDOR"].dropna().sort_values("  GDOR", ascending=False)
	OTHPER_df = df[df["Class"] == "OTHPER"].dropna().sort_values("OTHPER", ascending=False)
	Noise_df  = df[df["Class"] == " Noise"].dropna().sort_values(" Noise", ascending=False)

	# XXX - DO NOT DELETE THESE
	# This one below is used for training SOM_clean_1.csv
	# INTERIM_SOM
#	thresholds = np.array([0.6, 0.7, 0.85, 0.85, 0.65, 0.95, 0.85])

	# This one below is used for training RF_clean_1.csv
	# INTERIM_RF
#	thresholds = np.array([0.6, 0.6, 0.9, 0.89, 0.66, 0.985, 0.927])

	# This one below is used for k2_1-4_training_set_download.sh and eventually
	# Training set ALPHA 
#	thresholds = np.array([0.6, 0.7, 0.9, 0.89, 0.66, 0.990, 0.981])
	# Training set BETA 
#	thresholds = np.array([0.5, 0.62, 0.86, 0.81, 0.56, 0.988, 0.95])
	# Training Set Gamma Provisional
#	thresholds = np.array([0.4, 0.5, 0.5, 0.5, 0.5, 0.95, 0.85])
	# Training Set POINT4
#	thresholds = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
	# TRAINING SET POINT5
	thresholds = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

	# Choose Probability Thresholds here
	RRab_df   =   RRab_df[  RRab_df["  RRab"] > thresholds[0]]
	EA_df     =     EA_df[    EA_df["    EA"] > thresholds[1]]
	EB_df     =     EB_df[    EB_df["    EB"] > thresholds[2]]
	DSCUT_df  =  DSCUT_df[ DSCUT_df[" DSCUT"] > thresholds[3]]
	GDOR_df   =   GDOR_df[  GDOR_df["  GDOR"] > thresholds[4]]
	OTHPER_df = OTHPER_df[OTHPER_df["OTHPER"] > thresholds[5]]
	Noise_df  =  Noise_df[ Noise_df[" Noise"] > thresholds[6]]
	
	# If in the future this needs adding to, only add from campaign_3 and campaign_4
	full_set = RRab_df.append(EA_df).append(EB_df).append(DSCUT_df).append(GDOR_df)\
	                      .append(OTHPER_df).append(Noise_df).sort_values("epic_number")\
	                      [['epic_number', 'Campaign', 'Class', 'Probability']] 

	# XXX Very CLUNKY - Hopefully get to change at some point.
	# Had to previoulsy exclude these 8 as k2sc detrending wouldnt work
#	problem_set = np.array([201167905, 201280924, 201686483, 201720674, 201873161, 201908974, 204558226, 205170167])
	problem_set = np.array([])

	training_df = full_set[(full_set['epic_number'].isin(gaia_epics))]

	print(training_df)

	print('Training Set Break Down')
	print("  RRab: ", len(training_df[training_df['Class'] == '  RRab']))
	print("    EA: ", len(training_df[training_df['Class'] == '    EA']))
	print("    EB: ", len(training_df[training_df['Class'] == '    EB']))
	print(" DSCUT: ", len(training_df[training_df['Class'] == ' DSCUT']))
	print("  GDOR: ", len(training_df[training_df['Class'] == '  GDOR']))
	print("OTHPER: ", len(training_df[training_df['Class'] == 'OTHPER']))
	print(" Noise: ", len(training_df[training_df['Class'] == ' Noise']))

	output_file = "{}/training_sets/{}/c1-4_{}.csv"\
	               .format(project_dir, detrending, training_label)

	if write_to_csv:
		training_df.to_csv("{}".format(output_file), index=False)
		print("Written to file.")


def make_manual_k2sc_download_script(write_to_csv=False):

	# Open Master Table
	data_file = "{}/known/armstrong_0_to_4.csv".format(px402_dir)
	df   = pd.read_csv(data_file)
	df['Probability'] = df[['  RRab', '    EA', '    EB', ' DSCUT', '  GDOR', 'OTHPER', ' Noise']].max(axis=1)

	# Restrict to campaigns only 1, 2, 3, 4
	df = df[df["Campaign"] != 0]
#	df_34 = df[df["Campaign"].isin([3, 4])][["epic_number", "Class", "Campaign", "Probability"]]

	RRab_df   = df[df["Class"] == "  RRab"].dropna().sort_values("  RRab", ascending=False)
	EA_df     = df[df["Class"] == "    EA"].dropna().sort_values("    EA", ascending=False)
	EB_df     = df[df["Class"] == "    EB"].dropna().sort_values("    EB", ascending=False)
	DSCUT_df  = df[df["Class"] == " DSCUT"].dropna().sort_values(" DSCUT", ascending=False)
	GDOR_df   = df[df["Class"] == "  GDOR"].dropna().sort_values("  GDOR", ascending=False)
	OTHPER_df = df[df["Class"] == "OTHPER"].dropna().sort_values("OTHPER", ascending=False)
	Noise_df  = df[df["Class"] == " Noise"].dropna().sort_values(" Noise", ascending=False)

	# Initial set of thresholds
	# This one below is used for k2_1-4_training_set_download.sh 
	thresholds = np.array([0.6, 0.7, 0.9, 0.89, 0.66, 0.990, 0.981])

	# Choose Probability Thresholds here
	RRab_df   =   RRab_df[  RRab_df["  RRab"] > thresholds[0]]
	EA_df     =     EA_df[    EA_df["    EA"] > thresholds[1]]
	EB_df     =     EB_df[    EB_df["    EB"] > thresholds[2]]
	DSCUT_df  =  DSCUT_df[ DSCUT_df[" DSCUT"] > thresholds[3]]
	GDOR_df   =   GDOR_df[  GDOR_df["  GDOR"] > thresholds[4]]
	OTHPER_df = OTHPER_df[OTHPER_df["OTHPER"] > thresholds[5]]
	Noise_df  =  Noise_df[ Noise_df[" Noise"] > thresholds[6]]
	
#	t_set = RRab_df.append(EA_df).append(EB_df).append(DSCUT_df).append(GDOR_df)\
#	                      .append(OTHPER_df).append(Noise_df).sort_values("epic_number")\
#	                      ['epic_number'].to_numpy()
#	t_set = RRab_df.append(EA_df).append(EB_df).append(DSCUT_df).append(GDOR_df)\
#	                      .append(OTHPER_df).append(Noise_df).sort_values("epic_number")\
#	                      [['epic_number', 'Class', 'Campaign', 'Probability']]
#	t_set = t_set[t_set["Campaign"].isin([1, 2])]
#	all_available_epics_df = t_set.append(df_34, ignore_index=True)
#	all_available_epics_df.to_csv("here.csv", index=False)

	# Now produce script to download these epic in k2 raw data.
	script_file = "{}/scripts/antares_k2_1-4_download.sh".format(project_dir)
	with open(script_file, 'a') as ofile:
		for entry in t_set.to_numpy():
			epic = str(entry[0])
			campaign_num = entry[1]

			command = "curl -O https://archive.stsci.edu/missions/k2/lightcurves/c{}/{}00000/{}000/ktwo{}-c0{}_llc.fits"\
			            .format(campaign_num, epic[0:4], epic[4:6], epic, campaign_num)
			ofile.write("{}\n".format(command))

	output_file = "{}/training_sets/1-4_training_set.csv".format(project_dir)

	if (write_to_csv):
		t_set.to_csv("{}".format(output_file), index=False)


def change_training_set_labels(detrending='k2sc'):
	point5_df = pd.read_csv("{}/training_sets/{}/c1-4_point5.csv".format(project_dir, detrending))
	standalone_rrab_df = pd.read_csv("{}/training_sets/{}/c1-4_point4.csv".format(project_dir, detrending)) # O.4 RRab that we are eventually including
	standalone_rrab_df = standalone_rrab_df[standalone_rrab_df['epic_number'] == 210681941]
	print(point5_df)
	print(standalone_rrab_df)
	original_train_df = point5_df.append(standalone_rrab_df, ignore_index=True)
	print(original_train_df)
#	changed_epics_cjh_df = pd.read_csv("{}/training_sets/{}/manual_changes_cjh.csv".format(project_dir, detrending))
#	changed_epics_sjh_df = pd.read_csv("{}/training_sets/{}/manual_changes_sjh.csv".format(project_dir, detrending))
#	changed_epics_df = changed_epics_cjh_df.append(changed_epics_sjh_df, ignore_index=True)
	changed_epics_df = pd.read_csv("{}/training_sets/k2sc/final_decision_1.csv".format(project_dir))
	changed_epics = changed_epics_df['epic_number'].to_numpy()
	print(changed_epics_df)

	# Copy
	output_df = original_train_df
	output_df.columns = ['epic_number', 'Campaign', 'DJA_Class', 'DJA_Probability']

	# Make loop to change every entry in the file
	new_classes = []
	for i, epic in enumerate(output_df['epic_number']):
		if epic in changed_epics:
			new_class = changed_epics_df[changed_epics_df['epic_number'] == epic].iloc[0]['final_decision']
			new_class = np.nan if new_class == 'REMOVE' else new_class
			new_classes.append(new_class)
		else:
			armstrong_class = output_df[output_df['epic_number'] == epic].iloc[0]['DJA_Class']
			new_classes.append(armstrong_class)
	print(new_classes)

	output_df['Class'] = new_classes
	print(output_df)
	output_df = output_df.dropna()
	output_df.to_csv("{}/training_sets/k2sc/c1-4_delta.csv".format(project_dir), index=False)
	return

def get_training_set_stats():
	tset_file = "{}/training_sets/k2sc/c1-4_delta.csv".format(project_dir)
	train_df = pd.read_csv(tset_file) 

	orig_RRab = len(train_df[train_df['DJA_Class'] == '  RRab'])
	new_RRab = len(train_df[train_df['Class'] == '  RRab'])
	orig_EA = len(train_df[train_df['DJA_Class'] == '    EA'])
	new_EA = len(train_df[train_df['Class'] == '    EA'])
	orig_EB = len(train_df[train_df['DJA_Class'] == '    EB'])
	new_EB = len(train_df[train_df['Class'] == '    EB'])
	orig_GDOR = len(train_df[train_df['DJA_Class'] == '  GDOR'])
	new_GDOR = len(train_df[train_df['Class'] == '  GDOR'])
	orig_DSCUT = len(train_df[train_df['DJA_Class'] == ' DSCUT'])
	new_DSCUT = len(train_df[train_df['Class'] == ' DSCUT'])
	orig_OTHPER = len(train_df[train_df['DJA_Class'] == 'OTHPER'])
	new_OTHPER = len(train_df[train_df['Class'] == 'OTHPER'])
	orig_Noise = len(train_df[train_df['DJA_Class'] == ' Noise'])
	new_Noise = len(train_df[train_df['Class'] == ' Noise'])

	print("Original Number of   RRab: {}".format(orig_RRab))
	print("     New Number of   RRab: {}\n".format(new_RRab))
	print("Original Number of     EA: {}".format(orig_EA))
	print("     New Number of     EA: {}\n".format(new_EA))
	print("Original Number of     EB: {}".format(orig_EB))
	print("     New Number of     EB: {}\n".format(new_EB))
	print("Original Number of  DSCUT: {}".format(orig_DSCUT))
	print("     New Number of  DSCUT: {}\n".format(new_DSCUT))
	print("Original Number of   GDOR: {}".format(orig_GDOR))
	print("     New Number of   GDOR: {}\n".format(new_GDOR))
	print("Original Number of  Noise: {}".format(orig_Noise))
	print("     New Number of  Noise: {}\n".format(new_Noise))
	print("Original Number of OTHPER: {}".format(orig_OTHPER))
	print("     New Number of OTHPER: {}\n".format(new_OTHPER))

def main():
#	make_manual_k2sc_download_script(write_to_csv=False)
#	make_training_set(detrending='k2sc', training_label='point5', write_to_csv=True)
#	change_training_set_labels()
	get_training_set_stats()

if __name__ == "__main__":
	main()
