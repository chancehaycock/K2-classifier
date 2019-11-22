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


def make_2D_SOM_from_lightcurve(campaign_num):
	known = campaign_is_known(campaign_num)

	# Get SOM samples
	df = pd.read_csv('{}/som_bins/campaign_{}_0_1_norm.csv'\
	                 .format(px402_dir, campaign_num), 'r', delimiter=',')
	# Remove any samples without complete entries.
	df = df.dropna()

	# Down Sampling SOM
	noise_sample_df = df[df["Class"] == "Noise"].sample(n=100, random_state=1)
	othper_sample_df = df[df["Class"] == "OTHPER"].sample(n=100, random_state=1)

	df = df.drop(df[df["Class"] == "Noise"].index)
	df = df.drop(df[df["Class"] == "OTHPER"].index)

	df = pd.concat([df, othper_sample_df, noise_sample_df])
	df = df.sort_index()

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
	n_bins = 64
	som_samples = df_without_class.to_numpy(dtype=np.float32)
	# Remove EPICs from table. (NOT Necessary for model)
	som_samples = np.delete(som_samples, [64], 1)

	def init(som_samples):
		return np.random.uniform(0, 2, size=(som_shape[0], som_shape[1], n_bins))

	# Initialise and Plot
	som = SimpleSOMMapper(som_shape, n_iter, initialization_func=init)
	som.train(som_samples)
	# TODO - PRINT THE KOHONEN LAYER!!!!!!
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
		plt.scatter(*zip(*EA), s=2.5, c='r', label='EA')
	if (len(EB) > 0):
		plt.scatter(*zip(*EB), s=2.5, c='b', label='EB')
	if (len(RRab) > 0):
		plt.scatter(*zip(*RRab), s=2.5, c='g', label='RRab')
	if (len(DSCUT) > 0):
		plt.scatter(*zip(*DSCUT), s=2.5, c='y', label='DSCUT')
	if (len(GDOR) > 0):
		plt.scatter(*zip(*GDOR), s=2.5, c='c', label='GDOR')
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
	plt.show()

	# TODO -
	# Generate SOM statistics here with the distances, and bmu etc.
	# Also do the binned statistics here.



def main():
#	make_2D_SOM_from_features(3)

	make_2D_SOM_from_lightcurve(3)

if __name__ == "__main__":
	main()
