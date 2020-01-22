# The aim of this script is to take in master_tables 3 and 4 and restrict them
# to a training set ready for use in all models.o

from kepler_data_utilities import *

def make_SOM_training_set(write_to_csv=False):

	# Open Master Tables 3 and 4
	data_file = "{}/known/armstrong_0_to_4.csv".format(px402_dir)
	df   = pd.read_csv(data_file)

	# Restrict to campaigns only 3 an 4
	df = df[((df["Campaign"] == 3) | (df["Campaign"] == 4))]

	RRab_df   = df[df["Class"] == "  RRab"].dropna().sort_values("  RRab", ascending=False)
	EA_df     = df[df["Class"] == "    EA"].dropna().sort_values("    EA", ascending=False)
	EB_df     = df[df["Class"] == "    EB"].dropna().sort_values("    EB", ascending=False)
	DSCUT_df  = df[df["Class"] == " DSCUT"].dropna().sort_values(" DSCUT", ascending=False)
	GDOR_df   = df[df["Class"] == "  GDOR"].dropna().sort_values("  GDOR", ascending=False)
	OTHPER_df = df[df["Class"] == "OTHPER"].dropna().sort_values("OTHPER", ascending=False)
	Noise_df  = df[df["Class"] == " Noise"].dropna().sort_values(" Noise", ascending=False)

	# Initial set of thresholds
	# This one below is used for training SOM_clean_1.csv
	thresholds = np.array([0.6, 0.7, 0.85, 0.85, 0.65, 0.95, 0.85])

	# Choose Probability Thresholds here
	RRab_df   =   RRab_df[  RRab_df["  RRab"] > thresholds[0]]
	EA_df     =     EA_df[    EA_df["    EA"] > thresholds[1]]
	EB_df     =     EB_df[    EB_df["    EB"] > thresholds[2]]
	DSCUT_df  =  DSCUT_df[ DSCUT_df[" DSCUT"] > thresholds[3]]
	GDOR_df   =   GDOR_df[  GDOR_df["  GDOR"] > thresholds[4]]
	OTHPER_df = OTHPER_df[OTHPER_df["OTHPER"] > thresholds[5]].head(len(RRab_df)+10)
	Noise_df  =  Noise_df[ Noise_df[" Noise"] > thresholds[6]].head(0)
	
	useful_epics = RRab_df.append(EA_df).append(EB_df).append(DSCUT_df).append(GDOR_df)\
	                      .append(OTHPER_df).append(Noise_df).sort_values("epic_number")\
	                      ['epic_number'].to_numpy()
	
	# Original SOM bins
	original_som3_file = "{}/som_bins/campaign_3.csv".format(project_dir)
	original_som4_file = "{}/som_bins/campaign_4.csv".format(project_dir)
	som3_df = pd.read_csv(original_som3_file)
	som4_df = pd.read_csv(original_som4_file)
	total_df = som3_df.append(som4_df, ignore_index=True)
	total_df = total_df[total_df['epic_number'].isin(useful_epics)].dropna()

	print(total_df)

	output_file = "{}/training_sets/c34_clean_1_RF.csv".format(project_dir)

	if write_to_csv:
		total_df.to_csv("{}".format(output_file), index=False)

	print(len(RRab_df))
	print(len(EA_df))
	print(len(EB_df))
	print(len(DSCUT_df))
	print(len(GDOR_df))
	print(len(OTHPER_df))
	print(len(Noise_df))


def make_RF_training_set(write_to_csv=False):

	# Open Master Tables 3 and 4
	data_file = "{}/known/armstrong_0_to_4.csv".format(px402_dir)
	df   = pd.read_csv(data_file)

	# Restrict to campaigns only 3 an 4
	df = df[((df["Campaign"] == 3) | (df["Campaign"] == 4))]

	RRab_df   = df[df["Class"] == "  RRab"].dropna().sort_values("  RRab", ascending=False)
	EA_df     = df[df["Class"] == "    EA"].dropna().sort_values("    EA", ascending=False)
	EB_df     = df[df["Class"] == "    EB"].dropna().sort_values("    EB", ascending=False)
	DSCUT_df  = df[df["Class"] == " DSCUT"].dropna().sort_values(" DSCUT", ascending=False)
	GDOR_df   = df[df["Class"] == "  GDOR"].dropna().sort_values("  GDOR", ascending=False)
	OTHPER_df = df[df["Class"] == "OTHPER"].dropna().sort_values("OTHPER", ascending=False)
	Noise_df  = df[df["Class"] == " Noise"].dropna().sort_values(" Noise", ascending=False)

	# Initial set of thresholds
	# This one below is used for training RF_clean_1.csv
	thresholds = np.array([0.6, 0.6, 0.9, 0.89, 0.66, 0.985, 0.927])

	# Choose Probability Thresholds here
	RRab_df   =   RRab_df[  RRab_df["  RRab"] > thresholds[0]]
	EA_df     =     EA_df[    EA_df["    EA"] > thresholds[1]]
	EB_df     =     EB_df[    EB_df["    EB"] > thresholds[2]]
	DSCUT_df  =  DSCUT_df[ DSCUT_df[" DSCUT"] > thresholds[3]]
	GDOR_df   =   GDOR_df[  GDOR_df["  GDOR"] > thresholds[4]]
	OTHPER_df = OTHPER_df[OTHPER_df["OTHPER"] > thresholds[5]].head(len(RRab_df))
	Noise_df  =  Noise_df[ Noise_df[" Noise"] > thresholds[6]]
	
	useful_epics = RRab_df.append(EA_df).append(EB_df).append(DSCUT_df).append(GDOR_df)\
	                      .append(OTHPER_df).append(Noise_df).sort_values("epic_number")\
	                      ['epic_number'].to_numpy()
	
	original_RF3_file = "{}/tables/campaign_3_master_table_with_som_data.csv".format(project_dir)
	original_RF4_file = "{}/tables/campaign_4_master_table_with_som_data.csv".format(project_dir)
	RF3_df = pd.read_csv(original_RF3_file)
	RF4_df = pd.read_csv(original_RF4_file)
	total_df = RF3_df.append(RF4_df, ignore_index=True)
	total_df = total_df[total_df['epic_number'].isin(useful_epics)]

	print(total_df)

	output_file = "{}/training_sets/c34_clean_1_RF.csv".format(project_dir)

	if write_to_csv:
		total_df.to_csv("{}".format(output_file), index=False)

	print(len(RRab_df))
	print(len(EA_df))
	print(len(EB_df))
	print(len(DSCUT_df))
	print(len(GDOR_df))
	print(len(OTHPER_df))
	print(len(Noise_df))


def make_manual_k2sc_training_set(write_to_csv=False):

	# Open Master Table
	data_file = "{}/known/armstrong_0_to_4.csv".format(px402_dir)
	df   = pd.read_csv(data_file)

	# Restrict to campaigns only 1, 2, 3, 4
	df = df[df["Campaign"] != 0]

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
	
	t_set = RRab_df.append(EA_df).append(EB_df).append(DSCUT_df).append(GDOR_df)\
	                      .append(OTHPER_df).append(Noise_df).sort_values("epic_number")\
	                      [['epic_number', 'Campaign']]
	
	print(t_set)

	output_file = "{}/training_sets/1-4_training_set.csv".format(project_dir)

	if (write_to_csv):
		t_set.to_csv("{}".format(output_file), index=False)

		# Now produce script to download these epic in k2 raw data.
		script_file = "{}/scripts/antares_k2_1-4_download.sh".format(project_dir)
		with open(script_file, 'a') as ofile:
			for entry in t_set.to_numpy():
				epic = str(entry[0])
				campaign_num = entry[1]

				command = "curl -O https://archive.stsci.edu/missions/k2/lightcurves/c{}/{}00000/{}000/ktwo{}-c0{}_llc.fits"\
				            .format(campaign_num, epic[0:4], epic[4:6], epic, campaign_num)
				ofile.write("{}\n".format(command))


def main():
#	make_SOM_training_set(write_to_csv=False)
#	make_RF_training_set(write_to_csv=True)
#	make_manual_k2sc_training_set(write_to_csv=True)

if __name__ == "__main__":
	main()
