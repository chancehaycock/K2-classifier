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

# Use this to toggle between remote/local data sets
use_remote = False

def get_hdul(use_remote, kepler_id):
	kepler_num = kepler_id[4:13]
	campaign_num = kepler_id[16:17]
	path_to_dir=""
	if (use_remote):
		path_to_dir = "/storage/astro2/phujzc/k2sc_data/campaign_{}".format(campaign_num)
	else:
		path_to_dir = "/Users/chancehaycock/dev/machine_learning/px402/k2sc_data/campaign_{}".format(campaign_num)
	return fits.open('{}/hlsp_k2sc_k2_llc_{}-c0{}_kepler_v2_lc.fits'.format(path_to_dir, kepler_num, campaign_num))

def get_lightcurve(hdul, lc_type):
	if (lc_type == "PDC"):
		lc_type_indx = 1
	elif (lc_type == "SAP"):
		lc_type_index = 2
	else:
		print("Invalid lightcurve type passed. Choose PDC or SAP.")
		return
	flux = hdul[lc_type_indx].data['flux']
	trend_t = hdul[lc_type_indx].data['trtime']
	# Note that this is the detrended data as described above.
	flux_c = flux + trend_t - np.median(trend_t)
	times = hdul[lc_type_indx].data['time']
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

# ==================== Main =======================

def main():
	# Open hdul object from local directory. (Example)
	hdul = get_hdul(use_remote, "K2SC228682327-c05")

	# Fetch times and observations from PDC flux.
	times, flux_c = get_lightcurve(hdul, "PDC")

	# Initial plot of lightcurve
	line_plot(times, flux_c, "title", "lc_test_plot")

	# Remove nans
	# TODO - Remove from current array as opposed to copying.
	cleaned_times, cleaned_flux = remove_nans(times, flux_c)
	frequency, power =  LombScargle(cleaned_times, cleaned_flux).autopower()

	# Lomb Scargle Plot
	line_plot(frequency, power, "Title", "ls_test_plot")

	# Phase Fold Test
	index = np.argmax(power)
	period = 1.0 / frequency[index]
	folded_times = [cleaned_times[i]%period for i in range(len(cleaned_times))]
	scatter_plot(folded_times, cleaned_flux, "", "phase_fold_test_plot")


if __name__ == "__main__":
	main()

