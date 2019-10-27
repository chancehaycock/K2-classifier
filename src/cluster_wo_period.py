from kepler_data_utilities import *
from selfsom import *
project_dir = "/Users/chancehaycock/dev/machine_learning/px402"

# Classes
# EA, EB, RRAB, DSCUT, GDOR, OTHPER, (NOISE)

def make_2D_SOM(campaign_num):
	known = campaign_is_known(campaign_num)

	# Get SOM samples
	df = pd.read_csv('{}/cross_match_data/gaia/campaign_{}_mixed_table.csv'\
	                 .format(project_dir, campaign_num), 'r', delimiter=',')
	# Remove any samples without complete entries.
	df = df.dropna()
	# Make copy of data frame without string entries (class) so that we can 
	# convert it to a numpy array
	df_without_class = df.drop("Class", axis=1)
	# Variables needed for SOM object
	# Size of Kohonen Layer. 40x40 for visual. 1600x1 for RF
	som_shape = [40, 40]
	# Chose at complete random
	n_iter = 20
	# Number of features
	n_bins = 13
	som_samples = df_without_class.to_numpy(dtype=np.float32)
	# Remove EPICs from table. (NOT Necessary for model)
	som_samples = np.delete(som_samples, [0], 1)
	#XXX Repeat Command to remove the period column.
	som_samples = np.delete(som_samples, [0], 1)

	def init(som_samples):
		return np.random.uniform(0, 2, size=(som_shape[0], som_shape[1], n_bins))

	# Initialise and Plot
	som = SimpleSOMMapper(som_shape, n_iter, initialization_func=init)
	som.train(som_samples)
	map = som(som_samples)
	map = [[float(map[i][j]) + np.random.normal(0, 0.3) for j in range(2)]\
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
	for i in range(len(map)):
		test_object = df.iloc[i]['Class'].split()[0]
		if (test_object == 'EA'):
			EA.append(map[i])
		elif (test_object == 'EB'):
			EB.append(map[i])
		elif (test_object == 'RRab'):
			RRab.append(map[i])
		elif (test_object == 'GDOR'):
			GDOR.append(map[i])
		elif (test_object == 'OTHPER'):
			OTHPER.append(map[i])
		elif (test_object == 'DSCUT'):
			DSCUT.append(map[i])
		elif (test_object == 'Noise'):
			Noise.append(map[i])
		else:
			print("Error - no such class.")
	plt.scatter(*zip(*EA), s=1.5, c='r')
	plt.scatter(*zip(*EB), s=1.5, c='b')
	plt.scatter(*zip(*RRab), s=1.5, c='g')
	plt.scatter(*zip(*DSCUT), s=1.5, c='y')
	plt.scatter(*zip(*GDOR), s=1.5, c='c')
	plt.scatter(*zip(*OTHPER), s=0.1, c='m', alpha=0.3)
	print("Total classified stars: ", len(EA) + len(EB) + len(RRab)\
	                                + len(DSCUT) + len(GDOR))
	plt.xlabel("SOM X Pixel")
	plt.ylabel("SOM Y Pixel")
	plt.show()


def main():
	make_2D_SOM(4)

if __name__ == "__main__":
	main()
