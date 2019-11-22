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

def main():
	test_epics = [205898099, 205905261, 205906121, 205908778, 205910844, 205912245,
                  205926404, 205940923, 205941422]

if __name__ == "__main__":
	main()
