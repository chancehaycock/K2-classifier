from kepler_data_utilities import *
from selfsom import *

# Classes
# EA, EB, RRAB, DSCUT, GDOR, OTHPER, (NOISE)

def make_2D_SOM_from_features(campaign_num):
	known = campaign_is_known(campaign_num)

	# Get SOM samples
	df = pd.read_csv('{}/tables/campaign_{}_master_table_2_periods_ratio_test.csv'\
	                 .format(px402_dir, campaign_num), 'r', delimiter=',')
	# Remove any samples without complete entries.
	df = df.dropna()
	# Also remove any rows with their main period = 20.
	df = df[(df['Period_1'] != 20.0) & (df['Period_2'] != 20.0)]
	print(df)
	# Make copy of data frame without string entries (class) so that we can 
	# convert it to a numpy array
	if known:
		df_without_class = df.drop("Class", axis=1)
	# Variables needed for SOM object
	# Size of Kohonen Layer. 40x40 for visual. 1600x1 for RF
	som_shape = [40, 40]
	# Chose at complete random
	n_iter = 50
	# Number of features (Will be number of bins of phase folded lightcurve)
	n_bins = len(df.columns) - 2
	som_samples = df_without_class.to_numpy(dtype=np.float32)
	# Remove EPICs from table. (NOT Necessary for model)
	som_samples = np.delete(som_samples, [0], 1)

	def init(som_samples):
		return np.random.uniform(0, 2, size=(som_shape[0], som_shape[1], n_bins))

	# Initialise and Plot
	som = SimpleSOMMapper(som_shape, n_iter, initialization_func=init)
	som.train(som_samples)
	map = som(som_samples)
	float_map = [[float(map[i][j]) + np.random.normal(0, 0.3) for j in range(2)]\
	                                                   for i in range(len(map))] 
	EA = []
	EB = []
	RRab = []
	DSCUT = []
	GDOR = []
	OTHPER = []
	Noise = []
	# TODO Maybe use colour map instead.
	# Function?
	for i in range(len(float_map)):
		test_object = df.iloc[i]['Class'].split()[0]
		if (test_object == 'EA'):
			EA.append(float_map[i])
		elif (test_object == 'EB'):
			EB.append(float_map[i])
		elif (test_object == 'RRab'):
			RRab.append(float_map[i])
		elif (test_object == 'GDOR'):
			GDOR.append(float_map[i])
		elif (test_object == 'OTHPER'):
			OTHPER.append(float_map[i])
		elif (test_object == 'DSCUT'):
			DSCUT.append(float_map[i])
		elif (test_object == 'Noise'):
			Noise.append(float_map[i])
		else:
			print("Error - no such class.")
	plt.scatter(*zip(*EA), s=1.5, c='r', label='EA')
	plt.scatter(*zip(*EB), s=1.5, c='b', label='EB')
	plt.scatter(*zip(*RRab), s=1.5, c='g', label='RRab')
	plt.scatter(*zip(*DSCUT), s=1.5, c='y', label='DSCUT')
	plt.scatter(*zip(*GDOR), s=1.5, c='c', label='GDOR')
	plt.scatter(*zip(*OTHPER), s=0.1, c='m', alpha=0.3)
	print("Total classified stars: ", len(EA) + len(EB) + len(RRab)\
	                                + len(DSCUT) + len(GDOR))
	plt.xlabel("SOM X Pixel")
	plt.ylabel("SOM Y Pixel")
	plt.legend()
	plt.show()


def make_2D_SOM_from_lightcurve(campaign_num, plot_kohonen=False, plot_som=False):

	known = campaign_is_known(campaign_num)

	# ======================================
	#                 TRAIN
	# ======================================
	# Requires a nice clean training set.
	# We will do this by restricting this set to ones with sufficiently large
	# probabilties.

	# Need to first import the training set. Do that here and call it..
#	training_df = pd.read_csv("{}/training_sets/c3_training_set_30_each.csv"\
#	                           .format(project_dir))
	# Just in case!!
#	training_df = training_df.dropna()

	# =====================================
	#        IMPORT NEW UKNOWN DATA
	# =====================================
	print("Importing SOM bins file...")
	# Get SOM samples
#	df = pd.read_csv('{}/som_bins/campaign_{}.csv'\
#	                 .format(px402_dir, campaign_num), 'r', delimiter=',')

#==============================================================================
#==============================================================================

	# =========================
	#    Training 1 - 30 Each
	# =========================
#	df = pd.read_csv("{}/training_sets/c3_training_set_30_each.csv"\
#	                 .format(project_dir))

	# =========================
	#    Training 2 - 50 Each (100 Noise and OTHPER)
	# =========================
#	df = pd.read_csv("{}/training_sets/c3_training_set_50_each_100_noise_othper.csv"\
#	                 .format(project_dir))

	# =========================
	#    Training 3 - 50 Each (250 Noise and OTHPER)
	# =========================
#	df = pd.read_csv("{}/training_sets/c3_training_set_50_each_250_noise_othper.csv"\
#	                 .format(project_dir))

	# =========================
	#    Training 4 - Probability1 
	# =========================
#	df = pd.read_csv("{}/training_sets/c3_training_set_probability1.csv"\
#	                 .format(project_dir))

	# =========================
	#  Training 5 C3 AND C4 Top 500 
	# =========================
#	df = pd.read_csv("{}/training_sets/c34_top500_probability.csv"\
#	                 .format(project_dir))

	# =========================
	#  Training 6 C3 AND C4 Top 1000 
	# =========================
	df = pd.read_csv("{}/training_sets/c34_top1000_probability.csv"\
	                 .format(project_dir))

#==============================================================================
#==============================================================================

	# The useful dataframe for SOM statistics.
	df = df.dropna()

	# Make copy of data frame without string entries (class) so that we can 
	# convert it to a numpy array
	if known:
		df_without_class = df.drop("Class", axis=1).drop("Probability", axis=1)

	# ===============================
	# Variables needed for SOM object
	# ===============================

	# Size of Kohonen Layer. 40x40 for visual. 1600x1 for RF
	som_shape = [40, 40]

	# Chose at complete random
	n_iter = 25

	# Number of features (Will be number of bins of phase folded lightcurve)
	n_bins = 64

	# Convert df to 2D numpy array
	som_samples = df_without_class.to_numpy(dtype=np.float32)

	# Remove EPICs from table. NOT Necessary for model
	som_samples = np.delete(som_samples, [64], 1)

	def init(som_samples):
		return np.random.uniform(0, 2, size=(som_shape[0], som_shape[1], n_bins))

	# Initialise and Plot
	som = SimpleSOMMapper(som_shape, n_iter, initialization_func=init)

	# XXX This line should change to train(training_samples)
	print("Training SOM...")
	som.train(som_samples)
	print("SOM Trained")

	if (plot_kohonen):

		# Get Final Kohonen Layer
		final_kohonen = som._access_kohonen()
		# Plot Kohonen Layer
		print("Setting Up Axes")
		fig, axs = plt.subplots(8, 8, sharex=True, sharey=True,
		                        gridspec_kw={'hspace': 0, 'wspace':0})
		print("Axes Set up.")
		x = np.linspace (0, 1, n_bins)
		for i in range(0, som_shape[0], 5):
			for j in range(0, som_shape[1], 5):
				redi = int(i/5)
				redj = int(j/5)
				# Rotation 90 degrees anticlockwise due to matplotlib axes 
				# convention. Now SOM and Kohonen layers can be compared.
				axi = -redj % int(som_shape[0]/5)
				axj = redi
				axs[axi, axj].scatter(x, final_kohonen[i][j], s=0.75)
				axs[axi, axj].set_ylim(0,1)
				axs[axi, axj].set_yticklabels([])
				axs[axi, axj].set_xticklabels([])
		plt.subplots_adjust(top=0.9)
		fig.suptitle("Kohonen Layer - 8x8 filter of 40x40")
		plt.savefig("{}/plots/training_set_candidates/c34_top1000_kohonen.png".format(project_dir))
		plt.close()

	# XXX - At this point, SOM is trained, and there are templates setup. At 
	# this stage in the code, I would like to print these out.

	# Then at this point, we map the new sample or just call best matching pixel.
	# This MAP is now an array of triples (best_x, best_y, distance_to_them)
	print("Mapping samples to the SOM, and calculating bmus...")
	map = som(som_samples)
	print("BMU's calculated for whole sample.")

	# ===========================================
	#          PLOTTING OF KNOWN SOM
	# ===========================================

	if (plot_som):

		# This float_map adds random jitter to the co-ordinates
		# (first two args).
		float_map = [[float(map[i][j]) + np.random.normal(0, 0.5)\
		               for j in range(2)] for i in range(len(map))] 
		EA = []
		EB = []
		RRab = []
		DSCUT = []
		GDOR = []
		OTHPER = []
		Noise = []
		for i in range(len(float_map)):
			test_object = df.iloc[i]['Class'].split()[0]
			if (test_object == 'EA'):
				EA.append(float_map[i])
			elif (test_object == 'EB'):
				EB.append(float_map[i])
			elif (test_object == 'RRab'):
				RRab.append(float_map[i])
			elif (test_object == 'GDOR'):
				GDOR.append(float_map[i])
			elif (test_object == 'OTHPER'):
				OTHPER.append(float_map[i])
			elif (test_object == 'DSCUT'):
				DSCUT.append(float_map[i])
			elif (test_object == 'Noise'):
				Noise.append(float_map[i])
			else:
				print("Error - no such class.")
		if (len(EA) > 0):
			plt.scatter(*zip(*EA), s=3.5, c='r', label='EA')
		if (len(EB) > 0):
			plt.scatter(*zip(*EB), s=3.5, c='b', label='EB')
		if (len(RRab) > 0):
			plt.scatter(*zip(*RRab), s=3.5, c='g', label='RRab')
		if (len(DSCUT) > 0):
			plt.scatter(*zip(*DSCUT), s=3.5, c='y', label='DSCUT')
		if (len(GDOR) > 0):
			plt.scatter(*zip(*GDOR), s=3.5, c='c', label='GDOR')
		if (len(OTHPER) > 0):
			plt.scatter(*zip(*OTHPER), s=0.1, c='m', label='OTHPER')
		if (len(Noise) > 0):
			plt.scatter(*zip(*Noise), s=0.1, c='k', label="Noise")
		print("Total classified stars: ", len(EA) + len(EB) + len(RRab)\
		                                + len(DSCUT) + len(GDOR))
		print("Total Noise or OTHPER stares: ", len(Noise) + len(OTHPER))

		plt.xlabel("SOM X Pixel")
		plt.ylabel("SOM Y Pixel")
		plt.legend()
		plt.savefig("{}/plots/training_set_candidates/c34_top1000_som.png".format(project_dir))
		plt.close()

def main():
#	make_2D_SOM_from_features(3)

	make_2D_SOM_from_lightcurve(campaign_num=3, plot_kohonen=True, plot_som=True)

if __name__ == "__main__":
	main()
