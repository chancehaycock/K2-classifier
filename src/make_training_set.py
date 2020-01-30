# The aim of this script is to take in master_tables 3 and 4 and restrict them
# to a training set ready for use in all models.o

from kepler_data_utilities import *

def make_training_set(detrending, training_label, write_to_csv=False):

	# Open Known Files
	data_file = "{}/known/armstrong_0_to_4.csv".format(px402_dir)
	df   = pd.read_csv(data_file)
	df['Probability'] = df[['  RRab', '    EA', '    EB', ' DSCUT', '  GDOR', 'OTHPER', ' Noise']].max(axis=1)

	# Restrict to campaigns only 3 an 4
	df = df[df["Campaign"] != 0]

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

	# ======================#
	#         ALPHA         #
	# ======================#
	# This one below is used for k2_1-4_training_set_download.sh and eventually
	# Training set ALPHA 
	thresholds = np.array([0.6, 0.7, 0.9, 0.89, 0.66, 0.990, 0.981])

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
	problem_set = np.array([201167905, 201280924, 201686483, 201720674, 201873161, 201908974, 204558226, 205170167])

#	t_set =  np.setdiff1d(full_set, problem_set)
	training_df = full_set[~full_set['epic_number'].isin(problem_set)]

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
	                      ['epic_number'].to_numpy()
	
	# Now produce script to download these epic in k2 raw data.
	script_file = "{}/scripts/antares_k2_1-4_download.sh".format(project_dir)
	with open(script_file, 'a') as ofile:
		for entry in t_set.to_numpy():
			epic = str(entry[0])
			campaign_num = entry[1]

			command = "curl -O https://archive.stsci.edu/missions/k2/lightcurves/c{}/{}00000/{}000/ktwo{}-c0{}_llc.fits"\
			            .format(campaign_num, epic[0:4], epic[4:6], epic, campaign_num)
			ofile.write("{}\n".format(command))

	print(t_set)

	output_file = "{}/training_sets/1-4_training_set.csv".format(project_dir)

	if (write_to_csv):
		t_set.to_csv("{}".format(output_file), index=False)



def main():
#	make_manual_k2sc_download_script(write_to_csv=True)
	make_training_set(detrending='k2sc', training_label='alpha', write_to_csv=True)

if __name__ == "__main__":
	main()
