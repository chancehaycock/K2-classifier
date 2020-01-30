from kepler_data_utilities import *
from selfsom import *

#==============================================================================
#                        SOM UTILITY FUNCTIONS
#==============================================================================

def get_training_samples(project_dir, detrending, training_file):

	print("Importing training file: {}.csv".format(training_file))

	# Import Training Epics
	training_epics_df = pd.read_csv('{}/training_sets/{}/{}.csv'\
	       .format(project_dir, detrending, training_file), 'r', delimiter=',')

	# Get rid of noise and make array of useful epics
	training_epics = training_epics_df[training_epics_df['Class'] != ' Noise']['epic_number'].to_numpy()

	# Import ALL known phasefolded bins from MASTER table
	bins_1 = pd.read_csv('{}/tables/{}/campaign_1_master_table.csv'.format(project_dir, detrending))
	bins_2 = pd.read_csv('{}/tables/{}/campaign_2_master_table.csv'.format(project_dir, detrending))
	bins_3 = pd.read_csv('{}/tables/{}/campaign_3_master_table.csv'.format(project_dir, detrending))
	bins_4 = pd.read_csv('{}/tables/{}/campaign_4_master_table.csv'.format(project_dir, detrending))

	# Merge and Reduce
	phasefold_df = bins_1.append(bins_2).append(bins_3).append(bins_4)
	train_df = phasefold_df[phasefold_df['epic_number'].isin(training_epics)]
	needed_columns = make_bin_columns(64)
	needed_columns.append('epic_number')
	needed_columns.append('probability')
	needed_columns.append('class')

	train_df = train_df[[x for x in needed_columns]]
	train_df = train_df.dropna() # Some are empty from binning. Drop.
	train_df_without_class = train_df.drop("class", axis=1).drop("probability", axis=1)

	# Convert df to 2D numpy array
	train_samples = train_df_without_class.to_numpy(dtype=np.float32)

	# Remove EPICs from table. NOT Necessary for model. 64th column.
	train_samples = np.delete(train_samples, [64], 1)

	return train_df, train_samples

def get_som_samples(train_df, train_samples, test_campaign_num, detrending, training_file):
	if (test_campaign_num == -1):
		print("SOM will display the {} training set.".format(training_file))
		som_samples_df = train_df
		som_samples = train_samples 
	else: 
		print("SOM will display the whole of campaign {} using the {} training set.".format(test_campaign_num, training_file))
		# Import test file
		som_samples_df = pd.read_csv('{}/tables/{}/campaign_{}_master_table.csv'\
		             .format(px402_dir, detrending, test_campaign_num), 'r', delimiter=',')
		needed_columns = make_bin_columns(64)
		needed_columns.append('epic_number')
		needed_columns.append('probability')
		needed_columns.append('class')
		som_samples_df = som_samples_df[[x for x in needed_columns]]
		# May need a drop NA?

		# Option to test with a known campaign.
		if (test_campaign_num in [1, 2, 3, 4]):
			# Remove Class and Prob from known campaigns.
			som_samples_df_without_class = som_samples_df.drop("class", axis=1).drop("probability", axis=1)
			# Convert to Numpy array
			som_samples = som_samples_df_without_class.to_numpy(dtype=np.float32)
		else:
			som_samples = som_samples_df.to_numpy(dtype=np.float32)

		# Remove EPICs from table. NOT Necessary for model. 64th column.
		som_samples = np.delete(som_samples, [64], 1)

	return som_samples_df, som_samples


def plot_kohonen_layer(som, n_bins, som_shape, save_plots, project_dir, kohonen_ofile):

	print("Plotting Kohonen Layer...")
	# Get Final Kohonen Layer
	final_kohonen = som._access_kohonen()
	# Plot Kohonen Layer
	print("Setting Up Axes")
	fig, axs = plt.subplots(8, 8, sharex=True, sharey=True,
	                        gridspec_kw={'hspace': 0, 'wspace':0})
	print("Axes set up.")
	x = np.linspace (0, 1, n_bins)
	pal = sbn.color_palette("Blues")
	for i in range(0, som_shape[0], 5):
		for j in range(0, som_shape[1], 5):
			redi = int(i/5)
			redj = int(j/5)
			# Rotation 90 degrees anticlockwise due to matplotlib axes 
			# convention. Now SOM and Kohonen layers can be compared.
			axi = -redj % int(som_shape[0]/5)
			axj = redi
			axs[axi, axj].set_ylim(0,1)
			axs[axi, axj].set_yticklabels([])
			axs[axi, axj].set_xticklabels([])
			axs[axi, axj].set_xticks([])
			axs[axi, axj].set_yticks([])
			df = pd.DataFrame(final_kohonen[i][j], index=x, columns=['points'])
			ax = sbn.scatterplot(x=df.index, y='points', s=2.0, hue='points',
			                     linewidth=0, data=df, ax=axs[axi, axj],
			                     c='b', legend=None)
			ax.set_ylabel('')    
			ax.set_xlabel('')


	plt.show()
	if save_plots:
		plt.tight_layout()
		plt.savefig("{}/plots/{}_kohonen_now.eps".format(project_dir, kohonen_ofile), format='eps')
#	plt.close()
	return None


def process_som_statistics(map, samples_df, som_shape, clusters, project_dir, test_campaign_num, detrending, training_file):
	som_stats = []
	for i, curve in enumerate(map):
		epic = samples_df.iloc[i]['epic_number']
		if np.isnan(curve[2]):
			som_stats.append([epic, np.nan, np.nan, np.nan, np.nan, np.nan])
			continue
		x_pixel = curve[0]
		y_pixel = curve[1]
		template_dist = curve[2]
		size_x = som_shape[0]
		size_y = som_shape[1]

		# Accounting for periodicity of the SOM
		left_cand_x  = x_pixel - size_x
		left_cand_y  = y_pixel

		right_cand_x = x_pixel + size_x
		right_cand_y = y_pixel

		up_cand_x    = x_pixel
		up_cand_y    = y_pixel + size_y

		down_cand_x  = x_pixel
		down_cand_y  = y_pixel - size_y

		leftup_cand_x = up_cand_x - size_x 
		leftup_cand_y = up_cand_y

		rightup_cand_x = up_cand_x + size_x
		rightup_cand_y = up_cand_y

		leftdown_cand_x = down_cand_x - size_x
		leftdown_cand_y = down_cand_y

		rightdown_cand_x = down_cand_x + size_x
		rightdown_cand_y = down_cand_y

		distances = []
		distances.append(epic)
		for cluster in clusters:
			norm_dist   = np.sqrt((x_pixel      - cluster[0])**2  + (y_pixel      - cluster[1])**2) 
			left_dist   = np.sqrt((left_cand_x  - cluster[0])**2  + (left_cand_y  - cluster[1])**2) 
			right_dist  = np.sqrt((right_cand_x - cluster[0])**2  + (right_cand_y - cluster[1])**2) 
			up_dist     = np.sqrt((up_cand_x    - cluster[0])**2  + (up_cand_y    - cluster[1])**2) 
			down_dist   = np.sqrt((down_cand_x  - cluster[0])**2  + (down_cand_y  - cluster[1])**2) 
			leftup_dist   = np.sqrt((leftup_cand_x  - cluster[0])**2  + (leftup_cand_y  - cluster[1])**2) 
			rightup_dist  = np.sqrt((rightup_cand_x - cluster[0])**2  + (rightup_cand_y - cluster[1])**2) 
			leftdown_dist   = np.sqrt((leftdown_cand_x  - cluster[0])**2  + (leftdown_cand_y  - cluster[1])**2) 
			rightdown_dist  = np.sqrt((rightdown_cand_x - cluster[0])**2  + (rightdown_cand_y - cluster[1])**2) 
			optimal_distance = np.nanmin([norm_dist, left_dist, right_dist, up_dist, down_dist,
			                              leftup_dist, rightup_dist, leftdown_dist, rightdown_dist])
			distances.append(optimal_distance)
		distances.append(template_dist)
		som_stats.append(distances)

	som_columns = ["epic_number", "RRab_dist", "EA_dist", "EB_dist", "GDOR_DSCUT_dist", "template_dist"]
	som_df = pd.DataFrame(som_stats, columns=som_columns) 
	som_df.to_csv('{}/som_statistics/{}/{}/campaign_{}.csv'.format(project_dir, detrending, training_file, test_campaign_num), index=False)
	return None

def plot_som(map, samples_df, som_shape, save_plots, project_dir, som_ofile):

	sbn.set(style="white")
	print("Plotting SOM")

	som_plot = []
	for i, curve in enumerate(map):
		epic = samples_df.iloc[i]['epic_number']
		sclass = samples_df.iloc[i]['class']
		prob = samples_df.iloc[i]['probability']
		if np.isnan(curve[2]):
			som_plot.append([epic, sclass, prob, np.nan, np.nan, np.nan])
			continue
		rand_x = np.mod(curve[0] + np.random.normal(0, 1.0), som_shape[0])
		rand_y = np.mod(curve[1] + np.random.normal(0, 1.0), som_shape[1])
		som_plot.append([epic, sclass, prob, rand_x, rand_y, curve[2]])
	som_plot_columns = ['epic', 'class', 'prob', 'float_x', 'float_y', 'temp_dist']
	som_plot_df = pd.DataFrame(som_plot, columns=som_plot_columns)

	# Drop Noise and OTHPER from the SOM plot.
	noise_rows = som_plot_df[som_plot_df['class'] == 'Noise'].index
	#othper_rows = som_plot_df[som_plot_df['class'] == 'OTHPER'].index
	not_class_rows = som_plot_df[som_plot_df['class'] == 'NOT CLASSIFIED'].index

	som_plot_df = som_plot_df.drop(noise_rows)\
	                         .drop(not_class_rows)

	palette2 = sbn.color_palette("husl", 6)
	ax = sbn.scatterplot(x='float_x', y='float_y', palette=palette2,
	                    linewidth=0,  hue='class', s=5, 
	                    alpha=0.5, data=som_plot_df, legend='brief')
	plt.xlabel("SOM X Pixel")
	plt.ylabel("SOM Y Pixel")
	plt.show()
	if save_plots:
		plt.tight_layout()
		print("Saving SOM File")
		plt.savefig("{}/plots/{}_som_now.eps".format(project_dir, som_ofile), format='eps')
#	plt.close()
	return None

def SOM_shape(dimension):
	som_shape = [0, 0]
	if (dimension == 1):
		som_shape[0] = 1600
		som_shape[1] = 1
	else:
		som_shape[0] = 40
		som_shape[1] = 40
	return som_shape


#==============================================================================
#                      SELF-ORGANISING MAP FUNCTION
#==============================================================================
# Returns dataframe of epics given in testfile with distances to nearest cluster
# and template distance. Also writes to csv.
def make_2D_SOM_from_lightcurve(test_campaign_num, training_file, dimension=2,
                                plot_kohonen=False, plot_SOM=False,
                                write_to_csv=False, save_plots=False, detrending='k2sc'):

	# ===============================
	# Variables needed for SOM object
	# ===============================

	# Size of SOM Map. 40x40 for visual. 1600x1 for RF
	som_shape = SOM_shape(dimension)

	# Chose at complete random - seems sufficient
	n_iter = 25

	# Number of features (Will be number of bins of phase folded lightcurve)
	n_bins = 64

	# ===============================
	#     Process Training Data
	# ===============================
	# Requires a nice clean training set.
	# We will do this by restricting this set to ones with sufficiently large
	# probabilties.
	train_df, train_samples = get_training_samples(project_dir, detrending, training_file)

	som_samples_df, som_samples = get_som_samples(train_df, train_samples, test_campaign_num, detrending, training_file)

	# Here, we have a som_samples array ready for mapping with 64 bins.
	print(som_samples_df)
	print(som_samples)

	# ===============================
	#      Initialise SOM Object 
	# ===============================

	# Initial Distribution in Kohonen Layer. Uniform.
	def init(som_samples):
		# Add seed for reproducible kohonen layer. Easier than saving it.
		np.random.seed(5)
		return np.random.uniform(0, 2, size=(som_shape[0], som_shape[1], n_bins))

	# Initialise SOM Object
	som = SimpleSOMMapper(som_shape, n_iter, initialization_func=init)

	# ===============================
	#         TRAIN The SOM
	# ===============================

	print("Training SOM...")
	som.train(train_samples)
	print("SOM Trained.")

	# ===============================
	# Option to Plot the Kohonen Layer
	# ===============================
	#  At this point, SOM is trained, and there are templates setup.
	# The user has the option to plot it here.

	if (plot_kohonen):
		plot_kohonen_layer(som, n_bins, som_shape, save_plots, project_dir, kohonen_ofile=training_file)


	# ===============================
	# Map test samples onto trained SOM
	# ===============================
	# Then at this point, we map the new sample to the best matching pixel.
	# This MAP is now an array of triples (best_x, best_y, distance_to_them)

	print("Mapping samples to the SOM, and calculating bmus...")
	map = som(som_samples)
	print("BMU's calculated for whole sample.")

	# ==========================
	#   Option to Plot SOM 
	# ==========================

	if (plot_SOM):
		plot_som(map, som_samples_df, som_shape, save_plots, project_dir, som_ofile=training_file)


	# ==========================
	#   Process SOM DISTANCES
	# ==========================
	# Now we process distances to return to the user as a csv file.
	# Need to return arry of len(som_samples) with entries
	# [rr_dist, ea_dist, eb_dist, gdor/dscut_dist, template_dist]

	# Judged by eye - using seed=5
	# RRab, EA, EB, GDOR/DSCUT
#	clusters = [[11, 13], [31, 21], [31, 6], [13, 33]] - INTERIM
	clusters = [[13, 29], [29, 29], [1, 29], [9, 9]] # alpha

	if write_to_csv:
		if test_campaign_num == -1:
			print("Invalid Test Campaign Number. Exiting...")
			return
		process_som_statistics(map, som_samples_df, som_shape, clusters, project_dir, test_campaign_num, detrending, training_file)

	# Return dataframe for possible later work.
	print("Program Complete.")
	return som_samples_df 


#==============================================================================
#                                  MAIN
#==============================================================================

def main():

	# ==================================
	# Training 1 C3 AND C4 Best 70/80 of each 
	# 0 Noise.
	# =================================
	training_interim = "c34_clean_1"

	alpha_training = "c1-4_alpha"

	# Make SOM
	# XXX -1 flag in test_campaign_num is a flag to plot the training set!
	# This needs to be done first to get the cluster centres.
	make_2D_SOM_from_lightcurve(test_campaign_num=3, training_file='c1-4_alpha',
	                            dimension=2, detrending='k2sc',
	                            plot_kohonen=True, plot_SOM=True,
	                            save_plots=False, write_to_csv=False)


if __name__ == "__main__":
	main()

