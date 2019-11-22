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
# I have found that these plots, along with the known campaign 3 classifications
# are good for checking that the prigrams are running in the intended manner.
def process_lcs_for_som(campaign_num, process_outliers=True, write_to_csv=False,
                        plot=False):
	# Adds columns to the exported csv. Only true on first iteration.
	add_bin_columns = True

	# Known star types (with useful calssifications)
	star_types = ["RRab", "DSCUT", "EA", "EB", "GDOR"]

	# Set number of bins for the phasefolded lightcurve
	n_bins = 64

	# Columns for the table
	columns = make_bin_columns(n_bins)

	data_file = '{}/tables/campaign_{}_master_table_flagged_on.csv'\
	            .format(px402_dir, campaign_num)
	df = pd.read_csv(data_file)

	with open('{}/som_bins/campaign_{}_0_1_norm.csv'\
	           .format(px402_dir, campaign_num), 'a+') as file:
		# Loop over all lightcurves in the campaign
		for i in range(len(df['epic_number'])):
			epic_num = int(df.iloc[i]['epic_number'])
			period = df.iloc[i]['Period_1']
			star_class = str(df.iloc[i]['Class'])

			# Restriction on the max period here. Any period greater than 20
			# does not have enough periodic information over the observation
			# period of 80 days. We therefore restrict our research to identified
			# periods of less than 20 days.
			if period < 20.0:
				# Processing of the lightcurve begins here
				hdul = get_hdul(epic_num, campaign_num)
				# By default chooses PDC
				times, flux = get_lightcurve(hdul, process_outliers=process_outliers)
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
				bin_means, bin_edges, binnumber = binned_statistic(phase,
				                                  normed_flux, 'mean', bins=n_bins)
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

				if (plot):
					if star_class.split()[0] in star_types:
						print("Plotting")
#						fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
#						# Standard K2SC lc Plot
#						ax1.plot(times, flux, linewidth=0.3)
#						# The fitted polynomial superposed
#						ax1.plot(times, polynomial, linewidth=1.5, c='m')
#						# K2SC lc - polyfit
#						ax2.plot(times, poly_flux, linewidth=0.3)
#						# Phase folded lightcurve
#						ax3.scatter(phase, normed_flux, s=0.2)
#						# Binned Lightcure here
#						ax4.scatter(bin_centres, bin_means, s=5)
#
						fig = plt.figure()
						gs = fig.add_gridspec(4, 4)
						ax1 = fig.add_subplot(gs[0, :])
						ax1.plot(times, flux, linewidth=0.3)
						ax1.plot(times, polynomial, linewidth=1.5, c='m')
						ax2 = fig.add_subplot(gs[1, :])
						ax2.plot(times, poly_flux, linewidth=0.3)
						ax3 = fig.add_subplot(gs[2:, :2])
						ax3.scatter(phase, normed_flux, s=0.2)
						ax4 = fig.add_subplot(gs[2:, 2:])
						ax4.scatter(bin_centres, bin_means, s=5)
						ax4.set_ylim([0, 1])

						fig.suptitle("Automated Period: {}".format(period))
						plot_dir = "{}/plots/phase_folds_4_way/"\
						           .format(px402_dir)
						plt.tight_layout()
						plt.subplots_adjust(top=0.85)
						plt.savefig("{}/{}_processing_plot_{}_c{}.png"\
						.format(plot_dir, star_class, epic_num, campaign_num))
						plt.close()
			else:
				# Nonsense. Gets filtered out at SOM creation stage anyway.
				bin_means = np.empty(n_bins) * np.nan 


			# ======================
			# Option to write to CSV
			# ======================
			star_class = star_class.split()[0]
			if (write_to_csv):
				row = pd.DataFrame(bin_means.reshape(-1, len(bin_means)),
				                   columns=columns)
				row['epic_number'] = epic_num
				row['Class'] = star_class.split()[0]
				if (add_bin_columns):
					add_bin_columns = False
					row.to_csv(file, index=None)
				else:
					row.to_csv(file, header=False, index=None)

			# Give the user a progress bar
			if i%50 == 0:
				print("{}%".format(float(100*i/16882)))
	# Confirmation of file compeletion
	print("{} created.".format(file))
	return None


# ========================================
#              MAIN PROGRAM
# ========================================

def main():
	print("Running...")
	process_lcs_for_som(3, process_outliers=True, write_to_csv=True, plot=True)

if __name__ == "__main__":
	main()
