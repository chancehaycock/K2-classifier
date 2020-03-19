# A script to create a csv file with all of the non-periodic statistics from
# the lightcurve. The amount of these statistical measures is not really limites
# and hence this script contains a lot more than is probably required.
# The ones used on armstrong et. al 2016 are as follows;
# 1) SOM_index
# 2) SOM_distance
# 4) Amplitude max-min of phase curve
# 3) p2p_98
# 4) p2p_mean
# 5) phase_p2p_max_binned
# 6) phase_p2p_mean
# 7) stddev_over_error


# Need to add extra statistics part to SOM part

from kepler_data_utilities import *
import scipy.stats

def lightcurve_statistics(campaign_num, detrending):

	# First need to import lightcurve and remove the fitted polynomial
	epics_file = '{}/periods/{}/campaign_{}.csv'\
	              .format(project_dir, detrending, campaign_num)
	epics_df = pd.read_csv(epics_file)

	epics_df = epics_df[['epic_number']]
	num_lcs = len(epics_df)

	binned_lc_df = pd.read_csv("{}/phasefold_bins/{}/campaign_{}_interpolated.csv".format(project_dir,
	                                                                detrending, campaign_num))
	columns = ["epic_number", "lc_amplitude", "p2p_98", "p2p_mean", "stddev",
	           "kurtosis", "skew", "iqr", "mad", "max_binned_p2p", "mean_binned_p2p"]

	df = pd.DataFrame(columns=columns)

	print("Calculating non-periodic statistics for {} {} lightcurves from campaign {}."\
	       .format(num_lcs, detrending, campaign_num))
	for i in range(len(epics_df)):

		# ===========================================
		#             Lightcurve Statistics 
		# ===========================================

		epic_num = int(epics_df.iloc[i])

		# CORRUPT FILE
		if epic_num == 206413104:
			continue 

		hdul = get_hdul(epic_num, campaign_num, detrending=detrending)
		# By default chooses PDC
		times, flux = get_lightcurve(hdul, process_outliers=True, detrending=detrending)

		flux_median = np.median(flux)
		flux /= flux_median

		# Add intermediate step here of fitting 3rd order polynomial
		# to remove long term periodic variations to help the phasefolds etc
		coefficients = np.polyfit(times,flux,3,cov=False)
		polynomial = np.polyval(coefficients,times)
		#subtracts this polynomial from the median divided flux
		poly_flux = flux - polynomial + 1

		# NOW DO ALL CALCULATIONS WITH POLY_FLUX
		# Amplitude
		amplitude = np.ptp(poly_flux)

		# Light Curve Point to Point
		p2p_diffs = np.diff(poly_flux)
		p2p_98    = np.percentile(p2p_diffs, 98)
		p2p_mean = np.mean(p2p_diffs)

		# Standard Deviation of Lightcurve
		stddev = np.std(poly_flux)

		# Exotic statistics
		kurtosis = scipy.stats.kurtosis(poly_flux)
		skew = scipy.stats.skew(poly_flux)

		# Interquartile range
		iqr = scipy.stats.iqr(poly_flux)

		# Median Absolute Deviation - Better with outliers
		mad = scipy.stats.median_absolute_deviation(poly_flux)

		# ===========================================
		#             Binned Statistics 
		# ===========================================
		# This could be done in process_lightcurve, but we do it here for
		# convenience and hence only create one extra csv file.

		binned_values = binned_lc_df[binned_lc_df["epic_number"] == epic_num]
		binned_values = binned_values.drop("epic_number", axis=1)

		diffs = np.diff(binned_values)
		# XXX Should this be absolute difference????
		max_binned_p2p = np.max(abs(diffs))
		mean_binned_p2p = np.mean(diffs)

		# Add to files here. Only Prints for now.
		df.loc[i] = np.zeros(len(columns))
		df.loc[i]["epic_number"]  = int(epic_num)
		df.loc[i]["lc_amplitude"] = amplitude
		df.loc[i]["p2p_98"]       = p2p_98
		df.loc[i]["p2p_mean"]     = p2p_mean
		df.loc[i]["stddev"]       = stddev
		df.loc[i]["kurtosis"]     = kurtosis
		df.loc[i]["skew"]         = skew
		df.loc[i]["iqr"]          = iqr
		df.loc[i]["mad"]          = mad
		df.loc[i]["max_binned_p2p"] = max_binned_p2p
		df.loc[i]["mean_binned_p2p"] = mean_binned_p2p

		# Give the user a progress bar
		if i%50 == 0:
			size = len(epics_df)
			print("{0:.2f}%".format(float(100*i/size)))

		if (False):
			print("\n==== Epic: {} ====".format(epic_num))
			print("Amplitude:\t%4.3f" % amplitude)
			print("p2p_98:   \t%4.3f" % p2p_98)
			print("p2p_mean  \t%4.3f" % p2p_mean)
			print("stddev    \t%4.3f" % stddev)
			print("Kurtosis  \t%4.3f" % kurtosis)
			print("Skew      \t%4.3f" % skew)
			print("IQR       \t%4.3f" % iqr)
			print("MAD       \t%4.3f" % mad)
			print("binp2pmax \t%4.3f" % max_binned_p2p)
			print("binp2pmean\t%4.3f\n" % mean_binned_p2p)

	print("Converting dataframe to csv file...")

	# Send Dataframe to CSV
	with open("{}/lightcurve_statistics/{}/campaign_{}.csv"\
	          .format(px402_dir, detrending, campaign_num), 'a+') as statfile:
		df.to_csv(statfile, index=False)
	print("Complete.")
	print("Statistics computed for {} objects.".format(size))

def main():
#	lc_statistics(3)
#	lc_statistics(4)
#	lightcurve_statistics(1, 'k2sc')
#	lightcurve_statistics(2, 'k2sc')
#	lightcurve_statistics(3, 'k2sc')
#	lightcurve_statistics(4, 'k2sc')
	lightcurve_statistics(5, 'k2sc')
	lightcurve_statistics(6, 'k2sc')
	lightcurve_statistics(7, 'k2sc')
	lightcurve_statistics(8, 'k2sc')
	lightcurve_statistics(10, 'k2sc')


if __name__ == "__main__":
	main()




