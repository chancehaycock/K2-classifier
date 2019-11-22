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

def lc_statistics(campaign_num):
	# First need to import lightcurve and remove the fitted polynomial
	epics_file = '{}/periods/campaign_{}_flagged_on.csv'\
	              .format(project_dir, campaign_num)
	epics_df = pd.read_csv(epics_file)
	epics_df = epics_df[['epic_number']]

	columns = ["epic_number", "lc_amplitude", "p2p_98", "p2p_mean", "stddev",
	           "kurtosis", "skew", "iqr", "mad"]
	df = pd.DataFrame(columns=columns)

	for i in range(len(epics_df)):
		epic_num = int(epics_df.iloc[i])

		hdul = get_hdul(epic_num, campaign_num)
		# By default chooses PDC
		times, flux = get_lightcurve(hdul, process_outliers=True)
		flux_median = np.median(flux)

		# Add intermediate step here of fitting 3rd order polynomial
		# to remove long term periodic variations to help the phasefolds etc
		coefficients = np.polyfit(times,flux,3,cov=False)
		polynomial = np.polyval(coefficients,times)
		#subtracts this polynomial from the median divided flux
		poly_flux = flux-polynomial+flux_median

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

		# Give the user a progress bar
		if i%50 == 0:
			print("{}%".format(float(100*i/16882)))

		if (False):
			print("\n==== Epic: {} ====".format(epic_num))
			print("Amplitude:\t%4.3f" % amplitude)
			print("p2p_98:   \t%4.3f" % p2p_98)
			print("p2p_mean  \t%4.3f" % p2p_mean)
			print("stddev    \t%4.3f" % stddev)
			print("Kurtosis  \t%4.3f" % kurtosis)
			print("Skew      \t%4.3f" % skew)
			print("IQR       \t%4.3f" % iqr)
			print("MAD       \t%4.3f\n" % mad)

#	print(df)
	print("Converting dataframe to csv file...")
	# Send Dataframe to CSV
	with open("{}/non-periodic_statistics/lightcurve_statistics_c{}.csv"\
	          .format(px402_dir, campaign_num), 'a+') as statfile:
		df.to_csv(statfile)
	print("Complete.")


def main():
	lc_statistics(3)

if __name__ == "__main__":
	main()




