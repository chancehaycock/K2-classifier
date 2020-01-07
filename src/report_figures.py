from kepler_data_utilities import *

def plot():
	dscut = 206032188
#	eb = 
#	rrab =
#	gdor =  
	process_outliers = True
	campaign_num = 3
	epics = [206032188, 206397568, 206202136, 206382857]



	fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
	

	periods_file = '{}/periods/campaign_{}.csv'.format(project_dir,
	                                                   campaign_num)
	df = pd.read_csv(periods_file)
	df = df[['epic_number', 'Period_1']]


	for i, epic in enumerate(epics):

		period_df = df[df['epic_number'] == epic]
		period = period_df.iloc[0]['Period_1']

		# Processing of the lightcurve begins here
		hdul = get_hdul(epic, 3)
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

		jj = int(i % 2)
		ii = int((i - jj) / 2)
		print(ii)
		print(jj)


		axs[ii][jj].scatter(phase, normed_flux, s=0.05)
		axs[ii][jj].set_xticklabels([])
		axs[ii][jj].set_yticklabels([])
		axs[ii][jj].set_xticks([])
		axs[ii][jj].set_yticks([])
	plt.tight_layout()
	plt.subplots_adjust(wspace=0, hspace=0)
	plt.savefig("lightcurve_report_plot.eps", format='eps')


def main():
	plot()


if __name__ == "__main__":
	main()


