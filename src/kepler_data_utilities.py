### The seven columns of data per lightcurve:
# - quality  : Original K2 photometry pipeline quality flags
# - cadence  : observation cadences
# - time     : observation times
# - flux     : K2SC-detrended flux 
# - trend_p  : k2sc-estimated position-dependent trend
# - trend_t  : k2sc-estimated time-dependent trend
# - mflags   : k2sc outlier flags
# The PDC-MAP and SAP extensions are identical, the only difference being
# the flux for which the detrending was applied to.

# The trend_t and trend_p columns contain the time- and
# position-depedent trends, respectively. Their baseline levels are the
# same as those of the original (input) and detrended fluxes. To compute
# the full K2SC model including both time- and position-dependent
# trends, one must first subtract the median from one of them:

# trend_tot = trend_t + trend_p - median(trend_p)

# The detrended flux was obtained by subtracting the full K2SC model,
# including both systematics and stellar variability, from the input flux,
# but ensuring the median is unchanged:
 
# flux = input_flux - trend_tot + np.median(trend_tot).

# This detrended flux could in principle be used to perform a transit
# search. However, please be warned that the stellar variability model
# used by K2SC is intended to help model and remove the systematics as
# well as possible, it is not optimized for subsequent transit searches.
#
# Instead, we encourage users to compute the systematics-only corrected
# flux, which preserves the astrophysical variability, by adding the
# time-dependent trend back on (after correcting the median):
#
# flux_c = flux + trend_t - median(trend_t)

# This preserves stellar variability, and is useful for a wide range of
# astrophysical studies. A separate variability filtering step can then
# be used to detrend the data with a view to performing a transit
# search.
#
# The mflags columns is used to flag outliers and data points which
# should be treated with caution or were excluded from the fit for any
# reason. It is a 16-bit integer, with each bit having a specific
# meaning:
#
# - 2**0 : one of the K2 quality flags on
# - 2**1 : flare (reserved but not currently used)
# - 2**2 : transit (reserved but not currently used)
# - 2**3 : upwards outlier
# - 2**4 : downwards outlier
# - 2**5 : nonfinite flux
# - 2**6 : a periodic mask applied manually by k2sc (not used in this version)
#
# The primary header is a direct copy of the original MAST primary
# header. The following header keywords are stored by K2SC in each
# extension:
#
# - SPLITS: time(s) of reversal of the direction of roll angle
#   variations (corresponds to break-points in the systematics model)
# - CDPP1R: our estimate of the 6.5h Combined Differential Photometric
#   Precision (CDPP) in the raw data. CDPP estimates are computed
#   following Gilliland et al. (2011).
# - CDPP1T: our estimate of the 6.5h Combined Differential Photometric
#   Precision (CDPP) in the systematics-corrected data
# - CDPP1C: our estimate of the 6.5h Combined Differential Photometric
#   Precision (CDPP) in the systematics-corrected and detrended data
# - KER_NAME: name of GP covariance function used for variability
#   component
# - KER_PARS: names of parameters of GP convariance function
# - KER_EQN: equation of covariance function
# - KER_HPS1: best-fit value of the parameters of GP covariance function.

from astropy.io import fits
from astropy.stats import LombScargle
from astropy.table import Table
from scipy.stats import binned_statistic
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn

# Use this to toggle between remote/local data sets
use_remote = False
px402_dir = '/Users/chancehaycock/dev/machine_learning/px402'
project_dir = '/Users/chancehaycock/dev/machine_learning/px402'

# We want to remove K2 Quality flags (0), upwards outliers (3) and non-finite
# flux(5)
def delete_useless_flags(times, flux, mflags):
	del_array = []
	for i, flag in enumerate(mflags):

		# K2 Quality Flags
		fcond1 = 2**0 <= flag <= 2**(0 + 1) - 1
		# Upwards Outlier
		fcond2 = 2**3 <= flag <= 2**(3 + 1) - 1
		# Non-Finite Flux
		fcond3 = 2**5 <= flag <= 2**(5 + 1) - 1

		if fcond1 or fcond2 or fcond3:
			del_array.append(i)

	flux = np.delete(flux, del_array)
	times = np.delete(times, del_array)
	return times, flux

def get_hdul(epic_num, campaign_num, use_remote=False) :
	path_to_dir=""
	if (use_remote):
		path_to_dir = "/storage/astro2/phujzc/k2sc_data/campaign_{}"\
		              .format(campaign_num)
	else:
		path_to_dir = "{}/k2sc_data/campaign_{}".format(px402_dir, campaign_num)
	return fits.open('{}/hlsp_k2sc_k2_llc_{}-c0{}_kepler_v2_lc.fits'\
	                 .format(path_to_dir, epic_num, campaign_num))

def get_lightcurve(hdul, lc_type='PDC', process_outliers=True):
	if (lc_type == "PDC"):
		lc_type_indx = 1
	elif (lc_type == "SAP"):
		lc_type_index = 2
	else:
		print("Invalid lightcurve type passed. Choose PDC or SAP.")
		return

	flux = hdul[lc_type_indx].data['flux']
	trend_t = hdul[lc_type_indx].data['trtime']
	times = hdul[lc_type_indx].data['time']
	mflags = hdul[lc_type_indx].data['mflags']

	# Note that this is the detrended data as described above.
	flux_c = flux + trend_t - np.median(trend_t)

	# IS THIS SOMETHING WE WANT TO DO???
	if (process_outliers): 
		times, flux_c = delete_useless_flags(times, flux_c, mflags)

	return times, flux_c

def campaign_is_known(campaign_num):
	known_campaigns = [0, 1, 2, 3, 4]
	if campaign_num in known_campaigns:
		return True
	return False

def make_bin_columns(n_bins):
	columns = []
	for i in range(n_bins):
		columns.append("bin_{}".format(i+1))
	return columns

# Gets the estimate of the perdiod from K2SC directly. Uses Lomb Scargle also.
def k2sc_period_estimate(hdul, lc_type="PDC"):
	if lc_type == "SAP":
		return np.float64(hdul[2].header["KER_HPS1"].split()[3])
	elif lc_type == "PDC":
		return np.float64(hdul[1].header["KER_HPS1"].split()[3])
	print("Error - enter 'PDC' or 'SAP' for lightcurve type.")


#==============================================================================
#                        SOM UTILITY FUNCTIONS
#==============================================================================

def get_training_samples(project_dir, training_file):

	print("Importing training file: {}.csv".format(training_file))

	# Ensure no noise or nan's in trainings set.
	train_df = pd.read_csv('{}/training_sets/{}.csv'\
	                 .format(project_dir, training_file), 'r', delimiter=',')

	# Make copy of data frame without string entries (class) so that we can 
	# convert it to a numpy array
	train_df_without_class = train_df.drop("Class", axis=1).drop("Probability", axis=1)

	# Convert df to 2D numpy array
	train_samples = train_df_without_class.to_numpy(dtype=np.float32)

	# Remove EPICs from table. NOT Necessary for model. 64th column.
	train_samples = np.delete(train_samples, [64], 1)

	return train_df, train_samples

def get_som_samples(train_df, train_samples, campaign_num, test_file):
	if (test_file == None):
		som_samples_df = train_df
		som_samples = train_samples 
	else: 
		# Import test file
		som_samples_df = pd.read_csv('{}/{}{}.csv'\
	                 .format(px402_dir, test_file, campaign_num), 'r', delimiter=',')
		# Option to test with a known campaign.
		if (campaign_num in [3, 4]):
			# Remove Class and Prob from known campaigns.
			som_samples_df_without_class = samples_df.drop("Class", axis=1).drop("Probability", axis=1)
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
#			axs[axi, axj].scatter(x, final_kohonen[i][j], s=0.75)
			ax = sbn.scatterplot(x=df.index, y='points', s=2.0, hue='points',
			                     linewidth=0, data=df, ax=axs[axi, axj],
			                     #palette="GnBu",
			                     c='b', legend=None)
			ax.set_ylabel('')    
			ax.set_xlabel('')


#	plt.show()
	if save_plots:
		plt.tight_layout()
		plt.savefig("{}/plots/{}_kohonen_now.eps".format(project_dir, kohonen_ofile), format='eps')
	plt.close()
	return None


def process_som_statistics(map, samples_df, som_shape, clusters, project_dir, campaign_num):
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
	som_df.to_csv('{}/som_statistics/campaign_{}.csv'.format(project_dir, campaign_num), index=False)
	return None

def plot_som(map, samples_df, som_shape, save_plots, project_dir, som_ofile):

	sbn.set(style="white")
	print("Plotting SOM")

	som_plot = []
	for i, curve in enumerate(map):
		epic = samples_df.iloc[i]['epic_number']
		sclass = samples_df.iloc[i]['Class']
		prob = samples_df.iloc[i]['Probability']
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
	othper_rows = som_plot_df[som_plot_df['class'] == 'OTHPER'].index

	som_plot_df = som_plot_df.drop(noise_rows).drop(othper_rows)

#	palette1 = sbn.color_palette("cubehelix", 6)
#	palette2 = sbn.color_palette("hls", 5)
	palette2 = sbn.color_palette("husl", 5)
	ax = sbn.scatterplot(x='float_x', y='float_y', palette=palette2,
	                    linewidth=0,  hue='class', s=7.5, 
	                    alpha=1.0, style='class', data=som_plot_df, legend='brief')
	#plt.setp(ax.get_legend().get_texts(), fontsize='8') # for legend text
	#plt.setp(ax.get_legend().get_title(), fontsize='10') # for legend title
	plt.xlabel("SOM X Pixel")
	plt.ylabel("SOM Y Pixel")
#	plt.show()
	if save_plots:
		plt.tight_layout()
		print("Saving SOM File")
		plt.savefig("{}/plots/{}_som_now.eps".format(project_dir, som_ofile), format='eps')
	plt.close()
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







def main():
	test_epics = [205898099, 205905261, 205906121, 205908778, 205910844, 205912245,
                  205926404, 205940923, 205941422]

if __name__ == "__main__":
	main()
