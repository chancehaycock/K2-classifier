# The aim of this script is to take in master_tables 3 and 4 and restrict them
# to a training set ready for use in all models.o

from kepler_data_utilities import *

def filter_master_table(write_to_csv=False):
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
	thresholds = np.array([0.6, 0.7, 0.85, 0.85, 0.65, 0.95, 0.85])

	# Choose Probability Thresholds here
	RRab_df   =   RRab_df[  RRab_df["  RRab"] > thresholds[0]]
	EA_df     =     EA_df[    EA_df["    EA"] > thresholds[1]]
	EB_df     =     EB_df[    EB_df["    EB"] > thresholds[2]]
	DSCUT_df  =  DSCUT_df[ DSCUT_df[" DSCUT"] > thresholds[3]]
	GDOR_df   =   GDOR_df[  GDOR_df["  GDOR"] > thresholds[4]]
	OTHPER_df = OTHPER_df[OTHPER_df["OTHPER"] > thresholds[5]]
	Noise_df  =  Noise_df[ Noise_df[" Noise"] > thresholds[6]]
	
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

	output_file = "{}/training_sets/c34_probability2.csv".format(project_dir)

	if write_to_csv:
		with open(output_file, 'a+') as ofile:
			total_df.to_csv(ofile, index=False)


def main():
	filter_master_table(write_to_csv=True)

if __name__ == "__main__":
	main()
