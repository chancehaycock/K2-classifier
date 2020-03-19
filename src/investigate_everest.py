# A script to investigate the use of outlier flags in EVEREST data and how it
# affects the results obtained from the LS periodogram. We did something similar
# in the K2SC case, however there are many more available flags for EVEREST
# data.

from matplotlib.ticker import FormatStrFormatter
# =================================================================
#                    Available EVEREST Flags
# =================================================================

# Original Kepler Flags

# 00		Attitude tweak
# 01		Safe mode
# 02		Coarse point
# 03		Earth point
# 04		Zero crossing
# 05		Desaturation event
# 06		Argabrightening
# 07		Cosmic ray
# 08		Manual exclude
# 10		Sudden sensitivity dropout
# 11		Impulsive outlier
# 12		Argabrightening
# 13		Cosmic ray
# 14		Detector anomaly
# 15		No fine point
# 16		No data
# 17		Rolling band
# 18		Rolling band
# 19		Possible thruster firing
# 20		Thruster firing

# Additional EVEREST Flags
# 23		Data point is flagged in the raw K2 TPF
# 24		Data point is a NaN
# 25		Data point was determined to be an outlier
# 26		Not used
# 27		Data point is during a transit/eclipse 

from kepler_data_utilities import *

classes = ['RRab', 'EA', 'EB', 'GDOR', 'DSCUT']

def plot_k2sc_vs_everest():
	# By default plots campaigns 3 and 4
	line_labels = ["'RAW' K2SC", "Processed K2SC", "'Raw'EVEREST", "Processed EVEREST"]
	known_file = '{}/known/armstrong_0_to_4.csv'.format(project_dir)
	df = pd.read_csv(known_file)
	df = df[df['Campaign'] >= 3]

	# Loop over all lightcurves in the campaign
	for i in range(len(df['epic_number'])):
		epic_num = int(df.iloc[i]['epic_number'])

		class_row = df[df['epic_number'] == epic_num]
		if (not class_row.empty):
			star_class = class_row.iloc[0]["Class"]
			campaign_num = class_row.iloc[0]["Campaign"]
			probability = class_row.iloc[0][star_class]
			star_class = star_class.split()[0]
		else:
			# Some mismatch in data sets.
			star_class = ""
			probability = -1

		if probability < 0.5 or star_class not in classes:
			continue

		k2sc_hdul = get_hdul(epic_num, campaign_num, detrending='k2sc')
		ever_hdul = get_hdul(epic_num, campaign_num, detrending='everest')
		k2sc_times1, k2sc_flux1 = get_lightcurve(k2sc_hdul, process_outliers=False, detrending='k2sc')
		k2sc_times2, k2sc_flux2 = get_lightcurve(k2sc_hdul, process_outliers=True, detrending='k2sc')
		ever_times1, ever_flux1 = get_lightcurve(ever_hdul, process_outliers=False, detrending='everest')
		ever_times2, ever_flux2 = get_lightcurve(ever_hdul, process_outliers=True, detrending='everest')
		fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
		lc1 = ax1.scatter(k2sc_times1, k2sc_flux1, s=0.25, label='k2sc', color='red')
		lc2 = ax2.scatter(k2sc_times2, k2sc_flux2, s=0.25, label='k2sc', color='blue')
		lc3 = ax3.scatter(ever_times1, ever_flux1, s=0.25, label='everest (w/o outliers)', color='green')
		lc4 = ax4.scatter(ever_times2, ever_flux2, s=0.25, label='everest (w/ outliers)', color='orange')


		fig.legend([lc1, lc2, lc3, lc4],     # The line objects
           line_labels   # The labels for each line
           )
		fig.suptitle("{} - {} ({})".format(epic_num, star_class, probability))
		plt.subplots_adjust(top=0.9)
		plt.savefig("{}/antares_temp_plots/{}_{}.png".format(px402_dir, star_class, epic_num))
		plt.close()

		if (i%50 == 0):
			print("{0:.2f}% complete.".format(i * 100 / len(df['epic_number'])))


def k2sc_ever():
	known_file = '{}/known/armstrong_0_to_4.csv'.format(project_dir)
	df = pd.read_csv(known_file)
	epic_num = 206032188
	epic_num = 201180520
	epic_num = 206032188
	epic_num = 206096844
	epic_num = 206145148
	class_row = df[df['epic_number'] == epic_num]
	campaign_num=3

	k2sc_hdul = get_hdul(epic_num, campaign_num, detrending='k2sc')
	ever_hdul = get_hdul(epic_num, campaign_num, detrending='everest')
	k2sc_times1, k2sc_flux1 = get_lightcurve(k2sc_hdul, process_outliers=False, detrending='k2sc')
	k2sc_times2, k2sc_flux2 = get_lightcurve(k2sc_hdul, process_outliers=True, detrending='k2sc')
	ever_times1, ever_flux1 = get_lightcurve(ever_hdul, process_outliers=True, detrending='everest')

	pdc_flux = k2sc_hdul[1].data['flux']
	pdc_times = k2sc_hdul[1].data['time']
	k2sc1_median = np.nanmedian(k2sc_flux1)
	k2sc2_median = np.nanmedian(k2sc_flux2)
	pdc_median = np.nanmedian(pdc_flux)
	ever_median = np.nanmedian(ever_flux1)

	k2sc_flux1 /= k2sc1_median
	k2sc_flux2 /= k2sc2_median
	pdc_flux /= pdc_median
	ever_flux1 /= ever_median

	fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, gridspec_kw={'hspace': 0.0})

#	sbn.scatterplot(x='x',y='y', ax=ax1, data=pdc_df)
#	sbn.scatterplot(x='x',y='y', ax=ax1, data=k2sc1_df)
#	sbn.scatterplot(x='x',y='y', ax=ax2, data=k2sc2_df)
	ax1.scatter(pdc_times, pdc_flux, s=0.25, label='pdc', color='black')
#	ax1.plot(pdc_times, pdc_flux, linewidth=0.05, label='pdc', color='black', alpha=0.6)
#	ax1.scatter(k2sc_times1, k2sc_flux1, s=0.25, label='k2sc', color='red', alpha=0.8)
	ax2.scatter(k2sc_times2, k2sc_flux2, s=0.25, label='k2sc - outliers removed', color='blue', alpha=0.65)
#	ax2.plot(k2sc_times2, k2sc_flux2, linewidth=0.05, label='k2sc - outliers removed', color='blue', alpha=0.65)
	ax3.scatter(ever_times1, ever_flux1, s=0.25, label='k2sc - outliers removed', color='red', alpha=0.65)
#	ax3.plot(ever_times1, ever_flux1, linewidth=0.05, label='k2sc - outliers removed', color='red', alpha=0.65)

	
#	# Add intermediate step here of fitting 3rd order polynomial
#	# to remove long term periodic variations to help the phasefolds etc
#	coefficients = np.polyfit(k2sc_times2, k2sc_flux2, 3,cov=False)
#	polynomial = np.polyval(coefficients, k2sc_times2)
#	#subtracts this polynomial from the median divided flux
#	poly_flux = flux-polynomial+flux_median
#
#	ax2.plot(k2sc_times2, k2sc_flux2)
#
	#ax1.set_ylabel('PDC Relative Flux')
	ax2.set_ylabel('Relative Flux')
	#ax3.set_ylabel('EVEREST Relative Flux')
	ax3.set_xlabel("Time (BJD - 2454833)")
	ax3.yaxis.set_major_formatter(FormatStrFormatter('%1.3f'))


	#plt.tight_layout()
#	plt.show()
	plt.savefig('{}/final_report_images/ever_vs_k2sc_scatter.pdf'.format(project_dir), format='pdf')
	plt.close()

	
def k2sc_outlier_poly():
	known_file = '{}/known/armstrong_0_to_4.csv'.format(project_dir)
	df = pd.read_csv(known_file)
	epic_num = 206032188
	epic_num = 201180520
	epic_num = 206032188
	epic_num = 206096844
	epic_num = 206145148
	epic_num = 201222038
	campaign_num = 1
	class_row = df[df['epic_number'] == epic_num]

	k2sc_hdul = get_hdul(epic_num, campaign_num, detrending='k2sc')
	k2sc_times1, k2sc_flux1 = get_lightcurve(k2sc_hdul, process_outliers=False, detrending='k2sc')
	k2sc_times2, k2sc_flux2 = get_lightcurve(k2sc_hdul, process_outliers=True, detrending='k2sc')

	k2sc1_median = np.nanmedian(k2sc_flux1)
	k2sc2_median = np.nanmedian(k2sc_flux2)

	k2sc_flux1 /= k2sc1_median
	k2sc_flux2 /= k2sc2_median

#	# Add intermediate step here of fitting 3rd order polynomial
#	# to remove long term periodic variations to help the phasefolds etc
	coefficients = np.polyfit(k2sc_times2, k2sc_flux2, 3,cov=False)
	polynomial = np.polyval(coefficients, k2sc_times2)
	#subtracts this polynomial from the median divided flux
	poly_flux = k2sc_flux2-polynomial + 1

	fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, gridspec_kw={'hspace': 0.15}, figsize=(10, 5))

	ax1.scatter(k2sc_times1, k2sc_flux1, s=0.35, label='K2SC Detrended Lightcurve', color='black')
	ax2.scatter(k2sc_times2, k2sc_flux2, s=0.35, color='black')
	ax2.plot(k2sc_times2, polynomial, linewidth=2.5, label='3rd Order Polynomial', color='red', alpha=0.55)
	ax3.scatter(k2sc_times2, poly_flux, s=0.35, label='Final Lightcurve', color='black')

	print(len(k2sc_flux1))
	print(len(k2sc_flux2))
	outlier_times = []
	outlier_flux = []
	for i, point in enumerate(k2sc_flux1):
		if k2sc_times1[i] not in k2sc_times2 and not np.isnan(point):
			outlier_times.append(k2sc_times1[i])
			outlier_flux.append(point)
	print(outlier_times)
	print(outlier_flux)
	print()
	print(len(outlier_times))
	print(len(outlier_flux))
	ax2.scatter(outlier_times, outlier_flux, s=5.0, color='blue', alpha=0.6, marker='^', label='Outliers')



	
	#ax1.set_ylabel('K2SC Relative Flux')
	ax2.set_ylabel('Relative Flux')
	#ax3.set_ylabel('EVEREST Relative Flux')
	ax3.set_xlabel("Time (BJD - 2454833)")
	ax3.yaxis.set_major_formatter(FormatStrFormatter('%1.3f'))

	#ax1.legend(loc='lower right')
	#ax2.legend(loc='lower right')
	#ax3.legend(loc='lower right')

	ax1.set_ylim([0.95, 1.05])
	ax2.set_ylim([0.95, 1.05])
	ax3.set_ylim([0.95, 1.05])

	plt.tight_layout()
#	plt.show()
	plt.savefig('{}/final_report_images/k2sc_outlier_long.pdf'.format(project_dir), format='pdf')
	plt.close()





def main():
#	plot_k2sc_vs_everest()
	#k2sc_ever()
	k2sc_outlier_poly()

if __name__ == "__main__":
	main()






