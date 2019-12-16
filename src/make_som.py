from kepler_data_utilities import *
from selfsom import *

#==============================================================================
#                      SELF-ORGANISING MAP FUNCTION
#==============================================================================
# Returns dataframe of epics given in testfile with distances to nearest cluster
# and template distance. Also writes to csv.
def make_2D_SOM_from_lightcurve(campaign_num, training_file, test_file=None, dimension=2,
                                plot_kohonen=False, plot_SOM=False,
                                write_to_csv=False, save_plots=False):

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
	train_df, train_samples = get_training_samples(project_dir, training_file)

	som_samples_df, som_samples = get_som_samples(train_df, train_samples, campaign_num, test_file)

	# Here, we have a som_samples array ready for mapping with 64 bins.



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
		plot_kohonen_layer(som, n_bins, som_shape, save_plots, project_dir, "interim1")



	# ===============================
	# Map test samples onto trained SOM
	# ===============================
	# Then at this point, we map the new sample to the best matching pixel.
	# This MAP is now an array of triples (best_x, best_y, distance_to_them)

	print("Mapping samples to the SOM, and calculating bmus...")
	map = som(som_samples)
	print("BMU's calculated for whole sample.")



	# ==========================
	#   Process SOM DISTANCES
	# ==========================
	# Now we process distances to return to the user as a csv file.
	# Need to return arry of len(som_samples) with entries
	# [rr_dist, ea_dist, eb_dist, gdor/dscut_dist, template_dist]

	# Judged by eye - using clean_1.csv and seed=5
	# RRab, EA, EB, GDOR/DSCUT

	clusters = [[11, 13], [31, 21], [31, 6], [13, 33]]
	if write_to_csv:
		process_som_statistics(map, som_samples_df, som_shape, clusters, project_dir, campaign_num)



	# ==========================
	#   Option to Plot SOM 
	# ==========================

	if (plot_SOM):
		plot_som(map, som_samples_df, som_shape, save_plots, project_dir, "interim1")

	# Return dataframe for possible later work.
	print("Program Complete.")
	return som_samples_df 



#==============================================================================
#                                  MAIN
#==============================================================================

def main():

	# ==============================
	#  Training 1 C3 AND C4 Top 500 
	# ==============================
	training1 = "c34_top500_probability"

	# ==============================
	#  Training 2 C3 AND C4 Top 1000 
	# ==============================
	training2 = "c34_top1000_probability"

	# ==================================
	#  Training 3 C3 AND C4 0.5 and over 
	# =================================
	training3 = "c34_probability_over_half"

	# ==================================
	#  Training 4 C3 AND C4 0.5 and over 
	# Only 100 OTHPER and 0 Noise
	# =================================
	training4 = "c34_probability_over_half_100_OTHPER_0_Noise"

	# ==================================
	# Training 5 C3 AND C4 0.5 and over 
	# Only 200 OTHPER and 0 Noise
	# =================================
	training5 = "c34_probability_over_half_200_OTHPER_0_Noise"

	# ==================================
	# Training 6 C3 AND C4 0.5 and over 
	# Only 600 OTHPER and 0 Noise
	# =================================
	training6 = "c34_probability_over_half_600_OTHPER_0_Noise"

	# ==================================
	# Training 7 C3 AND C4 Best 70/80 of each 
	# 0 Noise.
	# =================================
	training7 = "c34_clean_1"

	# ==================================
	# Test against whole campaign
	# =================================
	test_file = "som_bins/campaign_"


	# Plot Training Set
	make_2D_SOM_from_lightcurve(campaign_num=4, training_file=training7,
	                            test_file=None, dimension=2,
	                            plot_kohonen=True, plot_SOM=True,
	                            save_plots=True, write_to_csv=False)

	# Export whole campaign to table from training set
#	make_2D_SOM_from_lightcurve(campaign_num=4, training_file=training7,
#	                            test_file=test_file, dimension=2,
#	                            plot_kohonen=False, plot_som=False,
#	                            save_plots=False, write_to_csv=True)

if __name__ == "__main__":
	main()

