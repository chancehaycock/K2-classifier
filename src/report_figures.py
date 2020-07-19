from kepler_data_utilities import *

def plot():
#	eb = 
#	rrab =
#	gdor =  
	process_outliers = True
	campaign_num = 3
	epics = [206382857, 206397568, 206202136, 205982900]
	epics = [206032188]



#	fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
	fig, axs = plt.subplots()
	

	periods_file = '{}/periods/k2sc/campaign_{}.csv'.format(project_dir,
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
		min_flux_index = np.nanargmin(normed_flux)
		normed_flux = np.array([normed_flux[(i+min_flux_index)%len(normed_flux)] for i in range(len(normed_flux))])

		n_bins=64

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
		print(bin_means)
		print(len(bin_means))

		jj = int(i % 2)
		ii = int((i - jj) / 2)
		print(ii)
		print(jj)

		scatter_df = pd.DataFrame(columns=['x', 'y'])
		scatter_df['x'] = phase
		scatter_df['y'] = normed_flux

		bin_df = pd.DataFrame(columns=['x', 'means'])
		bin_df['x'] = np.linspace(0.0, 1.0, 64)
		bin_df['means'] = bin_means

#		sbn.scatterplot(x='x', y='y', data=scatter_df, ax=axs[ii][jj], linewidth=0.0, s=1.5, palette='YlGnBu', hue='y', hue_norm=(-2, 1), legend=False)
		sbn.scatterplot(x='x', y='y', data=scatter_df, ax=axs, linewidth=0.0, s=3.0, palette='YlGnBu', hue='y', hue_norm=(-2, 1), legend=False)
		sbn.scatterplot(x='x', y='means', data=bin_df, ax=axs, s=50.0, color='k', linewidth=3.0, legend=False)
	#	axs[ii][jj].set_xlabel("")
	#	axs[ii][jj].set_ylabel("")

		axs.set_xlabel("Phase")
		axs.set_ylabel("Relative Flux")

#	fig.add_subplot(111, frame_on=False)
#	plt.tick_params(labelcolor="none", bottom=False, left=False)
#	plt.xlabel('Phase')
#	plt.ylabel("Relative Flux")

#	fig.text(0.51, 0.02, 'Phase', ha='center')
#	fig.text(0.02, 0.5, 'Relative Flux', va='center', rotation='vertical')
#	plt.subplots_adjust(wspace=0.07, hspace=0.07)
	sbn.despine()
	plt.tight_layout()
#	plt.savefig("lightcurve_report_plot.eps", format='eps')
#	plt.show()
	plt.savefig('{}/final_report_images/example_phasefoldtransparent.pdf'.format(project_dir), format='pdf', transparent=True)
	plt.close()

def main():
	plot()


if __name__ == "__main__":
	main()


