# This script contains a function to phase-fold a whole campaign of lightcurve,
# phase-fold it basd on periods obatined by SJH in periodogram research, and
# eventually bin the folded lc into 64 bins, ready for clustering by the
# the self-organising map. See selfsom.py

from kepler_data_utilities import *
import matplotlib.gridspec as gridspec

# Produces csv file of epicnumber, bin1, bin2, bin3, bin4.
# Also comes with the option to plot all lightcurves at the 4 steps of
# processing. i.e.
	# 1) K2SC lightcurve with fitted polynomial
	# 2) K2SC lightcurve - fitted polynomial to remove long term variations
	# 3) Phase folded version of the lightcurve
	# 4) The binned lightcurve

# Needs to be done from period file and crossmatched with the classifications
# from armstron et. al here.
# Formally was process_lcs_for_som
def phasefold(campaign_num, detrending, process_outliers=True, write_to_csv=False,
                        plot=False):

	# Is campaign known?
	known_campaign = campaign_is_known(campaign_num) 

	# Adds columns to the exported csv. Only true on first iteration.
	add_bin_columns = True

	# Known star types (with useful classifications)
	star_types = ["RRab", "DSCUT", "EA", "EB", "GDOR"]

	# Set number of bins for the phasefolded lightcurve
	n_bins = 64

	# Columns for the table
	columns = make_bin_columns(n_bins)

	# Do everything from period file NOT master table. (Not created yet)
	periods_file = '{}/periods/{}/campaign_{}.csv'.format(project_dir, detrending,
	                                                      campaign_num)
	df = pd.read_csv(periods_file)

	# XXX - Remove any entries tyhat DONT HAVE A PERIOD - eg. c3 206413104 FILE DOWNLOAD CORRUPTION
	# XXX NEEDS TO BE REMOVED BEFORE RUNNING ON OTHER CAMPAIGNS
	if campaign_num == 3:
		df = df[np.isfinite(df['Period_1'])]

	df = df[['epic_number', 'Period_1']]
	num_lcs = len(df['epic_number'])

	# Load this if needed later.
	if known_campaign:
		classes_file = '{}/known/armstrong_0_to_4.csv'.format(project_dir)
		classes_df = pd.read_csv(classes_file)


	print("Processing {} {} lightcurves".format(num_lcs, detrending))
	print("Beginning to phasefold and bin into {} bins.".format(n_bins))

	output_file = '{}/phasefold_bins/{}/campaign_{}.csv'.format(px402_dir, detrending, campaign_num)
	with open(output_file, 'a+') as file:
		# Loop over all lightcurves in the campaign
		for i in range(len(df['epic_number'])):

			size = len(df['epic_number'])
			epic_num = int(df.iloc[i]['epic_number'])
			period = df.iloc[i]['Period_1']

			# If known fetch class and it probability
			if known_campaign:
				# Fetch star class here
				class_row = classes_df[classes_df['epic_number'] == epic_num]
				if (not class_row.empty):
					star_class = class_row.iloc[0]["Class"]
					probability = class_row.iloc[0][star_class]
					star_class = star_class.split()[0]
				else:
					# Some mismatch in data sets.
					star_class = ""
					probability = -1

			# Processing of the lightcurve begins here
			hdul = get_hdul(epic_num, campaign_num, detrending=detrending)
			# By default chooses PDC
			times, flux = get_lightcurve(hdul, process_outliers=process_outliers,
			                             detrending=detrending)
			flux_median = np.median(flux)

			# Add intermediate step here of fitting 3rd order polynomial
			# to remove long term periodic variations to help the phasefolds etc
			coefficients = np.polyfit(times,flux,3,cov=False)
			polynomial = np.polyval(coefficients,times)
			#subtracts this polynomial from the median divided flux
			poly_flux = flux-polynomial+flux_median

			# Shift time axis back to zero
			times -= times[0]

			# Adjust Period to a phase between 0.0 and 1.0
			phase = (times % period) / period

			# Normalise lcurve so flux in [0.0, 1.0]
			min_flux = np.nanmin(poly_flux)
			normed_flux = poly_flux - min_flux
			max_flux = np.nanmax(normed_flux)
			normed_flux /= max_flux

			# XXX - WHY! Sort points on their phase???
			points = [(phase[i], normed_flux[i]) for i in range(len(phase))]
			folded_lc = [point for point in points if not np.isnan(point[1])]
			folded_lc.sort(key=lambda x: x[0])
			phase = [folded_lc[i][0] for i in range(len(folded_lc))]
			normed_flux = [folded_lc[i][1] for i in range(len(folded_lc))] 

			# Bin the lightcurve here!
			try:
				bin_means, bin_edges, binnumber = binned_statistic(phase,
			                                  normed_flux, 'mean', bins=n_bins)
			except ValueError:
				print("Binned Statistics Value Error: {}".format(epic))
			bin_width = bin_edges[1] - bin_edges[0]
			bin_centres = bin_edges[1:] - bin_width/2
			min_bin_val = np.nanmin(bin_means)
			min_bin_index = np.nanargmin(bin_means)
			bin_means = np.array([bin_means[(i + min_bin_index)%n_bins] \
			                                    for i in range(n_bins)])
			# Rescale to bins between 0 and 1.
			bin_means -= min_bin_val
			bin_means_max = np.nanmax(bin_means)
			bin_means /= bin_means_max


			if known_campaign:
				if (plot):
					if star_class in star_types:
						print("Plotting")

						fig = plt.figure()
						gs = fig.add_gridspec(4, 4)
						ax1 = fig.add_subplot(gs[0, :])
						# Standard K2SC lc Plot
						ax1.plot(times, flux, linewidth=0.3)
						# The fitted polynomial superposed
						ax1.plot(times, polynomial, linewidth=1.5, c='m')
						ax2 = fig.add_subplot(gs[1, :])
						# K2SC lc - polyfit
						ax2.plot(times, poly_flux, linewidth=0.3)
						ax3 = fig.add_subplot(gs[2:, :2])
						# Phase folded lightcurve
						ax3.scatter(phase, normed_flux, s=0.2)
						ax4 = fig.add_subplot(gs[2:, 2:])
						# Binned Lightcure here
						ax4.scatter(bin_centres, bin_means, s=5)
						ax4.set_ylim([0, 1])

						print(probability)
						fig.suptitle("Period: %.3f Prob: %.2f" % (period, probability))
						plot_dir = "{}/plots/phase_folds_4_way_w_probs"\
						           .format(px402_dir)
						plt.tight_layout()
						plt.subplots_adjust(top=0.85)
						#plt.show()
						plt.savefig("{}/{}_processing_plot_{}_c{}.png"\
						.format(plot_dir, star_class, epic_num, campaign_num))
						plt.close()

			# ======================
			# Option to write to CSV
			# ======================
			if (write_to_csv):
				# Adds rows of SOM Bins Applicable to all.
				row = pd.DataFrame(bin_means.reshape(-1, len(bin_means)),
				                   columns=columns)
				row['epic_number'] = epic_num
				# Add additional info for known stars
			#	if known_campaign:
			#		row['Class'] = star_class
			#		row['Probability'] = probability 
				if (add_bin_columns):
					add_bin_columns = False
					row.to_csv(file, index=None)
				else:
					row.to_csv(file, header=False, index=None)

			# Give the user a progress bar
			if i%50 == 0:
				print("{0:.2f}%".format(float(100*i/size)))

	# Confirmation of file compeletion
	print("{} created. It has {} entries.".format(output_file, size))

	print("Filling Missing Bins for Campaign {}".format(campaign_num))
	interpolate_df = fill_missing_bins(campaign_num, detrending, method='linear')
	interpolate_file = "{}/phasefold_bins/{}/campaign_{}_interpolated.csv".format(project_dir, detrending, campaign_num)
	interpolate_df.to_csv("{}".format(interpolate_file), index=False)

	# Confirmation of file compeletion
	print("{} created. It has {} entries.".format(interpolate_file, size))

	return None

def fill_missing_bins(campaign_num, detrending, method='linear'):
	input_df = pd.read_csv('{}/phasefold_bins/{}/campaign_{}.csv'.format(project_dir, detrending, campaign_num))
	epics = input_df['epic_number']
	df = input_df.drop('epic_number', axis=1)
	for col in df:
		df[col] = pd.to_numeric(df[col], errors='coerce')
	null_data = df[df.isnull().any(axis=1)]
	print(null_data)
	num_missing = len(null_data)
	print("Using method: {} to interpolate missing values from {} epics".format(method, num_missing))
	fixed_data = df.interpolate(method=method, axis=1, limit_direction='forward')
	fixed_data['epic_number'] = epics
	print('Interpolation Complete.')
	return fixed_data


# ========================================
#              MAIN PROGRAM
# ========================================

def main():
	print("Running...")
#	process_lcs_for_som(3, process_outliers=True, write_to_csv=True, plot=True)
#	process_lcs_for_som(3, process_outliers=True, write_to_csv=True, plot=False)
#	process_lcs_for_som(4, process_outliers=True, write_to_csv=True, plot=True)
#	process_lcs_for_som(4, process_outliers=True, write_to_csv=True, plot=False)

#	phasefold(1, 'k2sc', process_outliers=True, write_to_csv=True, plot=False)
#	phasefold(2, 'k2sc', process_outliers=True, write_to_csv=True, plot=False)
#	phasefold(3, 'k2sc', process_outliers=True, write_to_csv=True, plot=False)
#	phasefold(4, 'k2sc', process_outliers=True, write_to_csv=True, plot=False)
#	phasefold(1, 'k2sc', process_outliers=True, write_to_csv=True, plot=False)
	#phasefold(4, 'k2sc', process_outliers=True, write_to_csv=True, plot=False)

#	campaign_num=4
#	detrending='k2sc'
#	interpolate_df = fill_missing_bins(campaign_num, detrending, method='linear')
#	interpolate_file = "{}/phasefold_bins/{}/campaign_{}_interpolated.csv".format(project_dir, detrending, campaign_num)
#	interpolate_df.to_csv("{}".format(interpolate_file), index=False)


#	phasefold(5, 'k2sc', process_outliers=True, write_to_csv=True, plot=False)
#	phasefold(6, 'k2sc', process_outliers=True, write_to_csv=True, plot=False)
#	phasefold(7, 'k2sc', process_outliers=True, write_to_csv=True, plot=False)
#	phasefold(8, 'k2sc', process_outliers=True, write_to_csv=True, plot=False)
	phasefold(10, 'k2sc', process_outliers=True, write_to_csv=True, plot=False)


if __name__ == "__main__":
	main()
