# A script to investigate the use of outlier flags in EVEREST data and how it
# affects the results obtained from the LS periodogram. We did something similar
# in the K2SC case, however there are many more available flags for EVEREST
# data.

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



def main():
	plot_k2sc_vs_everest()

if __name__ == "__main__":
	main()






