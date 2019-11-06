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
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic

# Use this to toggle between remote/local data sets
use_remote = False
px402_dir = '/Users/chancehaycock/dev/machine_learning/px402'

def get_hdul(use_remote, epic_num, campaign_num):
	path_to_dir=""
	if (use_remote):
		path_to_dir = "/storage/astro2/phujzc/k2sc_data/campaign_{}"\
		              .format(campaign_num)
	else:
		path_to_dir = "{}/k2sc_data/campaign_{}".format(px402_dir, campaign_num)
	return fits.open('{}/hlsp_k2sc_k2_llc_{}-c0{}_kepler_v2_lc.fits'\
	                 .format(path_to_dir, epic_num, campaign_num))

def get_lightcurve(hdul, lc_type='PDC', delete_flags=True):
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

	# XXX - TODO CHange this perhaps to work with any flag number.
	# XXX - DELETE ALL ENTRIES WITH FLAGS!!!!
	# IS THIS SOMETHING WE WANT TO DO???
	if (delete_flags): 
		del_array = np.nonzero(mflags)
		flux_c = np.delete(flux_c, del_array)
		times = np.delete(times, del_array)

	return times, flux_c

def remove_nans(times, flux_c):
	cleaned_times = []
	cleaned_flux = []
	for i in range(len(times)):
		if (not np.isnan(flux_c[i])):
			cleaned_times.append(times[i])
			cleaned_flux.append(flux_c[i])
	return cleaned_times, cleaned_flux

def scatter_plot(times, flux, title, filename):
	plt.scatter(times, flux, s=0.1)
	plt.xlabel("Time")
	plt.ylabel("Flux")
	plt.title("{}".format(title))
	plt.savefig("{}.png".format(filename))
	plt.close()

def line_plot(frequency, power, title, filename):
	plt.plot(frequency, power, linewidth=0.5)
	plt.xlabel("Frequency")
	plt.ylabel("Power")
	plt.title("{}".format(title))
	plt.savefig("{}.png".format(filename))
	plt.close()

def campaign_is_known(campaign_num):
	known_campaigns = [0, 1, 2, 3, 4]
	if campaign_num in known_campaigns:
		return True
	return False

# Phasefolds a particular lightcurve. Returns an array tuples representing the 
# phase folded lc points in [0, 1] x [0, 1].
# The minimum point occurs at (0, 0).
def phase_fold_lightcurve(epic_num, campaign_num, period, plot=True, delete_flags=True):
	hdul = get_hdul(use_remote, epic_num, campaign_num)
	# By default chooses PDC
	times, flux = get_lightcurve(hdul, delete_flags=delete_flags)
	times -= times[0]
	# Adjust Period to a phase between 0.0 and 1.0
	phase = (times % period) / period
	# Normalise lcurve so flux in [0.0, 1.0]
	min_flux = np.nanmin(flux)
	normed_flux = flux - min_flux
	max_flux = np.nanmax(normed_flux)
	normed_flux /= max_flux

	# Translate curve so that minimum value occurs at (0, 0)
	phase_of_min_flux = phase[np.nanargmin(normed_flux)]
	phase = (phase - phase_of_min_flux) % 1.0

	# Plot!
	if (plot):
		print("\t {}".format(period))
		fig, (ax1, ax2) = plt.subplots(2, 1)
		ax1.plot(times, flux, linewidth=0.3)
		ax2.scatter(phase, normed_flux, s=0.2)
		plt.show()

	points = [(phase[i], normed_flux[i]) for i in range(len(phase))]
	folded_lightcurve = [point for point in points if not np.isnan(point[1])]
	folded_lightcurve.sort(key=lambda x: x[0])
	return folded_lightcurve

def make_bin_columns(n_bins):
	columns = []
	for i in range(n_bins):
		columns.append("bin_{}".format(i+1))
	return columns

# Maybe produce csv file of epicnumber, bin1, bin2, bin3, bin4.
# This step should probably involve the scaling and normalistaion of the
# lightcurve retreived form the previous step.
def process_lcs_for_som(campaign_num, del_flags=True):
#	test_array = [[206131351, 3, 0.604989832], [206143957, 3, 4.17346],
#	              [206134477, 3, 8.168424456], [206047180, 3, 13.89769166]]

	add_bin_columns = True
	n_bins = 64

	# Columns for the table
	columns = make_bin_columns(n_bins)

	data_file = '{}/tables/campaign_{}_master_table.csv'.format(px402_dir, campaign_num)
	df = pd.read_csv(data_file)

	with open('{}/som_bins/campaign_{}.csv'.format(px402_dir, campaign_num), 'a+') as file:
		# Loop over lightcurves
		for i in range(len(df['epic_number'])):
			epic_num = int(df.iloc[i]['epic_number'])
			period = df.iloc[i]['Period_1']
			star_class = df.iloc[i]['Class']

			if period < 20.0:
				folded_lc = phase_fold_lightcurve(epic_num, campaign_num, period,
				                                  plot=False, delete_flags=del_flags)
				phase = [folded_lc[i][0] for i in range(len(folded_lc))]
				flux = [folded_lc[i][1] for i in range(len(folded_lc))]
				bin_means, bin_edges, binnumber = binned_statistic(phase, flux,
				                                  'mean', bins=n_bins)
				bin_width = bin_edges[1] - bin_edges[0]
				bin_centres = bin_edges[1:] - bin_width/2
			else:
				bin_means = np.empty(64) * np.nan 

			row = pd.DataFrame(bin_means.reshape(-1, len(bin_means)), columns=columns)
			row['epic_number'] = epic_num
			row['Class'] = star_class

			if (add_bin_columns):
				add_bin_columns = False
				row.to_csv(file, index=None)
			else:
				row.to_csv(file, header=False, index=None)

			if i%50 == 0:
				print("{}%".format(float(100*i/16882)))

	print("{} created.".format(file))
	return None

# ==================== Main =======================

def main():
	print("Running...")

	# Examples of phase folding lightcurve from campaign 3
	# 1) RRab
#	print("RRab:")
#	print("\t Automated Period 1")
#	phase_fold_lightcurve(206131351, 3, 0.604989832)
	# 2) EA
#	print("EA:")
#	print("\t Automated Period 1")
#	print("\t\t Deleted Outliers")
#	phase_fold_lightcurve(206143957, 3, 3.692879976)
#	print("\t\t Outliers Remain")
#	phase_fold_lightcurve(206143957, 3, 3.692879976, delete_flags=False)
	# Second best period
#	print("\t Automated Period 2")
#	phase_fold_lightcurve(206143957, 3, 8.338349424)
	# Chance judge by eye
#	print("\t Eye Period with outliers")
#	phase_fold_lightcurve(206143957, 3, 4.25, delete_flags=False)
#	print("\t Second Spreadsheet Period")
#	phase_fold_lightcurve(206143957, 3, 4.17346)
	# 3) EB
#	print("EB:")
#	print("\t Automated Period 1")
#	phase_fold_lightcurve(206134477, 3, 8.168424456)
	# Second best - seems multiple of 3 off.
#	print("\t Automated Period 2")
#	phase_fold_lightcurve(206134477, 3, 12.7891238)
	# Chance judge by eye version
#	print("\t Eye Period with Outliers")
#	phase_fold_lightcurve(206134477, 3, 4.24, delete_flags=False)
	# 4) DSCUT
#	print("DSCUT:")
#	print("\t Automated Period 1")
#	phase_fold_lightcurve(206047180, 3, 13.89769166)
#	print("\t Automated Period 2")
#	phase_fold_lightcurve(206047180, 3, 15.94297007)
	# 4) GDOR
#	print("GDOR")
#	print("\t Automated Period 1")
#	phase_fold_lightcurve(205993244, 3, 0.829789837)

	process_lcs_for_som(3)

if __name__ == "__main__":
	main()

