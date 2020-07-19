# A script to contain all figures/stats to be used in the results section of
# the final report.

from kepler_data_utilities import *

model_name = 'som_and_rf'
training_set = 'delta'
model_number = 1

# Returns dataframe of containing all periods from campaigns 5 - 10 \ {9}
def collate_periods():
	period_c5 = pd.read_csv(  '{}/periods/k2sc/campaign_5.csv'.format(project_dir))
	period_c6 = pd.read_csv(  '{}/periods/k2sc/campaign_6.csv'.format(project_dir))
	period_c7 = pd.read_csv(  '{}/periods/k2sc/campaign_7.csv'.format(project_dir))
	period_c8 = pd.read_csv(  '{}/periods/k2sc/campaign_8.csv'.format(project_dir))
	period_c10 = pd.read_csv(  '{}/periods/k2sc/campaign_10.csv'.format(project_dir))

	periods = period_c5.append(period_c6,  ignore_index=True)\
	                   .append(period_c7,  ignore_index=True)\
	                   .append(period_c8,  ignore_index=True)\
	                   .append(period_c10, ignore_index=True)
	return periods


def plot_unknown(classes, df, periods):
	n_bins = 64
	for star_class in classes:
		class_group = df[df['Class'] == star_class]
		for epic in class_group['epic_number']:
			row = class_group[class_group['epic_number'] == epic]
			campaign_num = row.iloc[0]['Campaign']
			probability = row.iloc[0][star_class]
			period = periods[periods['epic_number'] == epic].iloc[0]['Period_1']

			hdul = get_hdul(epic, campaign_num)

			# By default chooses PDC
			times, flux = get_lightcurve(hdul)
			flux_median = np.median(flux)

			# Add intermediate step here of fitting 3rd order polynomial
			# to remove long term periodic variations to help the phasefolds etc
			coefficients = np.polyfit(times,flux,3,cov=False)
			polynomial = np.polyval(coefficients,times)

			# Subtracts this polynomial from the median divided flux
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

			fig.suptitle("Predicted Class {} Probability: {}".format(star_class, probability))
			plot_dir = "{}/plots/unknown/{}_{}_{}/{}"\
			           .format(px402_dir, model_name, training_set, model_number, star_class)
			plt.tight_layout()
			plt.subplots_adjust(top=0.85)
			plt.savefig("{}/{}_{}_c{}.png"\
			.format(plot_dir, probability, epic, campaign_num))
			plt.close()
	return

def probability_thresholds(stars_df, classes):
	classes.append('OTHPER')
	classes.remove('RRab')
	#classes.remove('EB')
	
	colours = ['b', 'g', 'c', 'y', 'm', 'r']
	fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0})

	for i, star_class in enumerate(classes):
		candidates = stars_df[stars_df['Class'] == star_class]
		probs = candidates[star_class].to_numpy()
#		plt.hist(probs, bins=100)
#		plt.title("{}".format(star_class))
#		plt.show()
		sbn.distplot(probs, bins=30, kde=False, color=colours[i], label=star_class, axlabel=False, norm_hist=True, ax=ax1,
		             hist_kws={"histtype": "step", "linewidth": 1.5,
		                             "alpha": 0.8,}) 
	
	ax1.legend(loc='upper left')

	big_classes = ['RRab']
	for i, star_class in enumerate(big_classes):
		candidates = stars_df[stars_df['Class'] == star_class]
		probs = candidates[star_class].to_numpy()
		sbn.distplot(probs, bins=30, kde=False, color=colours[5 + i], label=star_class, axlabel=False, norm_hist=True, ax=ax2,
		             hist_kws={"histtype": "step", "linewidth": 1.5,
		                             "alpha": 0.8,}) 
	
	ax2.legend(loc='upper left')
#	plt.xlabel('Classification Probability')
#	plt.ylabel('Count')
#	fig.text(0.5, 0.04, 'Classification Probability', ha='center', va='center')
#	fig.text(0.06, 0.5, 'Normalised Density', ha='center', va='center', rotation='vertical')
#	plt.show()

	fig.add_subplot(111, frame_on=False)
	plt.tick_params(labelcolor="none", bottom=False, left=False)

	plt.xlabel('Classification Probability')
	plt.ylabel("Normalised Density")
	plt.tight_layout()
	plt.savefig('{}/unseen_probability_distribution.png'.format(project_dir), format='png')
	plt.savefig('{}/unseen_probability_distribution.pdf'.format(project_dir), format='pdf')
	plt.close()


#def classifier_comparison():
#	# Order RF, ETC, ADA, GB
#	som1_scores = [0.85, 0.89, 0.87, 0.89]
#	som2_scores = [0.84, 0.88, 0.86, 0.88]
#	score_df = pd.DataFrame(columns=['Metric', 'SOM_dim', 'Classifier', 'Score'])
#	score_df.loc[0] = :
#	




feature_axis_labels = {'epic_number':'epic_number', 'Period_1':'Period / Days', 'Period_2': 'Period / Days',
                       'amp_ratio_21' : 'Period 2 / Period 1 Ratio', 'abs_magnitude' : 'Absolute Magnitude',
       'bp_rp' : 'B-R Colour Index', 'bp_g': 'B-G Colour Index', 'g_rp':'G-R Colour Index',
       'lc_amplitude': 'Relative Flux Amplitude', 'p2p_98':'p2p_98', 'p2p_mean':'p2p_mean', 'stddev':'Standard Deviation',
       'kurtosis':'Kurtosis', 'skew':'Skew', 'iqr':'IQR', 'mad':'MAD', 'max_binned_p2p':'Binned P2P Maximum', 'mean_binned_p2p':'Binned P2P Mean',
       'RRab_dist':'RRab Cluster Distance', 'EA_dist':'EA Cluster Distance',
       'EB_dist':'EB Cluster Distance', 'GDOR_DSCUT_dist':'GDOR/DSCUT Cluster Distance', 
       'template_dist': 'Best Matching Template Distance'}

def feature_distribution(stars_df, classes, test_df):
	classes.append('OTHPER')
	colours = ['b', 'g', 'c', 'y', 'm', 'r']
	# Restrict by Probabilities
	features = test_df.columns
	features = ['Period_1', 'lc_amplitude', 'skew']
	prob_threshold = 0.85
	for feature in features[0:1]:
		#f, axes = plt.subplots(2, 3, sharex=False, figsize=(10, 4))
		f, axes = plt.subplots(2, 3, sharex=False, figsize=(10, 4), gridspec_kw={'hspace':0.3, 'wspace':0.2} )
	#	f, axes = plt.subplots()
		#sbn.despine(left=True)
		for i, star_class in enumerate(classes):
			y = i % 3
			x = int((i - y) / 3)
			best_stars_df = stars_df[stars_df[star_class] >= prob_threshold]
			candidates = best_stars_df[best_stars_df['Class'] == star_class]['epic_number'].to_numpy()
			vals = test_df[test_df['epic_number'].isin(candidates)][feature]
			print(vals)
			sbn.distplot(vals, kde=False, color='b', ax=axes[x, y], label=star_class, axlabel=False) 
			sbn.despine()
			#axes[x, y].set_xlabel('{}'.format(feature_axis_labels[feature]))
			#axes[x, y].set_ylabel('Count')
			axes[x, y].set_xlabel("")
			axes[x, y].set_ylabel("")

#			sbn.distplot(vals, kde=False, color=colours[i], ax=axes, label=star_class, axlabel=False, norm_hist=True,
#			             hist_kws={"histtype": "step", "linewidth": 1.5,
#			                             "alpha": 0.6,}) 
		
#		plt.subplots_adjust()

#		plt.legend()

		f.add_subplot(111, frame_on=False)
		plt.tick_params(labelcolor="none", bottom=False, left=False)

		plt.xlabel('{}'.format(feature_axis_labels[feature]))
		plt.ylabel("Count")
		plt.tight_layout()
#		plt.show()
		plt.savefig('{}/final_report_images/{}_{}_transparent.pdf'\
		           .format(project_dir, feature, prob_threshold), format='pdf', transparent=True)
		plt.close()

def average_train_probabilities():
	classes = ['RRab', 'EA', 'EB', 'GDOR', 'DSCUT', 'OTHPER', 'Noise']
	train_df = pd.read_csv('{}/src/models/som_and_rf_delta/train.csv'.format(project_dir))
	print(train_df)
	print(train_df.columns)
	for star_class in classes:
		df = train_df[train_df['class'] == star_class]
		av_prob = np.mean(df['probability'])
		min_prob = np.min(df['probability'])
		max_prob = np.max(df['probability'])
		print(av_prob)
		print('{} & {}'.format(min_prob, max_prob))
	return


def training_feature_distribution():
	classes = ['RRab', 'EA', 'EB', 'GDOR', 'DSCUT', 'OTHPER']
	colours = ['b', 'g', 'c', 'y', 'm', 'r']

	train_df = pd.read_csv('{}/src/models/som_and_rf_delta/train.csv'.format(project_dir))

	features = ['Period_1', 'lc_amplitude']
	for feature in features[:1]:
		f, axes = plt.subplots(2, 3, sharex=False, figsize=(10, 4), gridspec_kw={'hspace':0.3, 'wspace':0.3} )
	#	f, axes = plt.subplots(figsize=(10,3))
		#sbn.despine(left=True)
		for i, star_class in enumerate(classes):
			y = i % 3
			x = int((i - y) / 3)
			vals = train_df[train_df['class'] == star_class][feature]
			print(vals)
			sbn.distplot(vals, kde=False, color='b', ax=axes[x, y], label=star_class, axlabel=True) 
#			axes[x, y].set_xlabel('{}'.format(feature_axis_labels[feature]))
#			axes[x, y].set_ylabel('Count')

		#	sbn.distplot(vals, kde=False, color=colours[i], ax=axes, label=star_class, axlabel=False, norm_hist=True,
		#	             hist_kws={"histtype": "step", "linewidth": 1.5,
		#	                             "alpha": 0.6,}) 
		

		

		f.add_subplot(111, frame_on=False)
		plt.tick_params(labelcolor="none", bottom=False, left=False)

		plt.xlabel('{}'.format(feature_axis_labels[feature]))
		plt.ylabel("Count")

		plt.subplots_adjust()
		plt.tight_layout()
		#plt.legend()

#		plt.show()
		plt.savefig('{}/final_report_images/training_distribution_{}_labelled.pdf'\

		           .format(project_dir, feature), format='pdf')
		plt.close()



def feature_dist_2(stars_df, classes, test_df):
	classes.append('OTHPER')
	prob_threshold = 0.85
	best_stars_df = pd.DataFrame(columns = stars_df.columns)
	for star_class in classes:
		best_stars_df = best_stars_df.append(stars_df[stars_df[star_class] >= prob_threshold])

	pal = sbn.cubehelix_palette(6, rot=-.25, light=.7)
	g = sbn.FacetGrid(best_stars_df, row="", column='Class', aspect=15, height=.5, palette=pal)

	print(best_stars_df)
	return



def get_GAIA_data():
	df = pd.DataFrame()
	for campaign_num in [1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
		df = df.append(pd.read_csv('{}/cross_match_data/gaia/unique_gaia_campaign_{}_data.csv'.format(project_dir, campaign_num), low_memory=False))
	df['abs_magnitude'] = 5.0 + df['phot_g_mean_mag']\
	                         - 5.0 * np.log10(df['r_est']) 
	df = df[['bp_rp', 'abs_magnitude']]
	print(df)
	print(df.dropna())
	return df.dropna()

def get_all_GAIA_data():
	df = pd.DataFrame()
	for campaign_num in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
		camp_df = pd.read_csv('{}/cross_match_data/gaia/unique_gaia_campaign_{}_data.csv'.format(project_dir, campaign_num), low_memory=False)
		camp_df['campaign_num'] = campaign_num
	#	df = df.append(pd.read_csv('{}/cross_match_data/gaia/unique_gaia_campaign_{}_data.csv'.format(project_dir, campaign_num), low_memory=False))
		df = df.append(camp_df)
	df['abs_magnitude'] = 5.0 + df['phot_g_mean_mag']\
	                         - 5.0 * np.log10(df['r_est']) 
	df = df[['epic_number', 'bp_rp', 'abs_magnitude', 'r_est', 'campaign_num']]
	#print(df)
	#print(df.dropna())
	return df.dropna()

def get_training_GAIA_data(train_df):
	training_epics = train_df['epic_number'].to_numpy()
	df = pd.DataFrame()
	for campaign_num in [1, 2, 3, 4]:
		df = df.append(pd.read_csv('{}/cross_match_data/gaia/unique_gaia_campaign_{}_data.csv'.format(project_dir, campaign_num), low_memory=False))
	df['abs_magnitude'] = 5.0 + df['phot_g_mean_mag']\
	                         - 5.0 * np.log10(df['r_est']) 
	df = df[df['epic_number'].isin(training_epics)]
	df = df[['bp_rp', 'abs_magnitude']]
	print(df)
	print(df.dropna())
	return df.dropna()


def HR_diagram(HR_data, stars_df=None, test_df=None):
#	if stars_df:
#	rrab_candidates = stars_df[(stars_df['Class'] == 'RRab') & (stars_df['RRab'] > 0.98)]['epic_number'].to_numpy()
#	ea_candidates = stars_df[(stars_df['Class'] == 'EA') & (stars_df['EA'] > 0.8)]['epic_number'].to_numpy()
#	eb_candidates = stars_df[(stars_df['Class'] == 'EB') & (stars_df['EB'] > 0.8)]['epic_number'].to_numpy()
#	dscut_candidates = stars_df[(stars_df['Class'] == 'DSCUT') & (stars_df['DSCUT'] > 0.8)]['epic_number'].to_numpy()
#	gdor_candidates = stars_df[(stars_df['Class'] == 'GDOR') & (stars_df['GDOR'] > 0.7)]['epic_number'].to_numpy()
#	othper_candidates = stars_df[(stars_df['Class'] == 'OTHPER') & (stars_df['OTHPER'] > 0.8)]['epic_number'].to_numpy()
	rrab_candidates = stars_df[(stars_df['Class'] == 'RRab') & (stars_df['RRab'] > 0.8)]['epic_number'].to_numpy()
	ea_candidates = stars_df[(stars_df['Class'] == 'EA') & (stars_df['EA'] > 0.9)]['epic_number'].to_numpy()
	eb_candidates = stars_df[(stars_df['Class'] == 'EB') & (stars_df['EB'] > 0.8)]['epic_number'].to_numpy()
	dscut_candidates = stars_df[(stars_df['Class'] == 'DSCUT') & (stars_df['DSCUT'] > 0.7)]['epic_number'].to_numpy()
	gdor_candidates = stars_df[(stars_df['Class'] == 'GDOR') & (stars_df['GDOR'] > 0.8)]['epic_number'].to_numpy()
	othper_candidates = stars_df[(stars_df['Class'] == 'OTHPER') & (stars_df['OTHPER'] > 0.8)]['epic_number'].to_numpy()
	rrab_df = test_df[test_df['epic_number'].isin(rrab_candidates)]
	rrab_df['class'] = 'RRab'
	ea_df = test_df[test_df['epic_number'].isin(ea_candidates)]
	ea_df['class'] = 'EA'
	eb_df = test_df[test_df['epic_number'].isin(eb_candidates)]
	eb_df['class'] = 'EB'
	dscut_df = test_df[test_df['epic_number'].isin(dscut_candidates)]
	dscut_df['class'] = 'DSCUT'
	gdor_df = test_df[test_df['epic_number'].isin(gdor_candidates)]
	gdor_df['class'] = 'GDOR'
	othper_df = test_df[test_df['epic_number'].isin(othper_candidates)]
	othper_df['class'] = 'OTHPER'

	general_candidates = stars_df['epic_number'].to_numpy()
	plot_df = test_df[test_df['epic_number'].isin(general_candidates)]

	fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharey=True,gridspec_kw={'wspace':0.07, 'hspace':0.07}, sharex=True, figsize=(11, 5))

	sbn.scatterplot(x="bp_rp", y="abs_magnitude", hue="bp_rp", palette="ch:s=1.0,r=0.1_r", hue_norm=(-1, 3), s=0.3, linewidth=0, legend=False, data=HR_data, ax=ax1)
	sbn.scatterplot(x="bp_rp", y="abs_magnitude", hue="bp_rp", palette="ch:s=1.0,r=0.1_r", hue_norm=(-1, 3), s=0.3, linewidth=0, legend=False, data=HR_data, ax=ax2)
	sbn.scatterplot(x="bp_rp", y="abs_magnitude", hue="bp_rp", palette="ch:s=1.0,r=0.1_r", hue_norm=(-1, 3), s=0.3, linewidth=0, legend=False, data=HR_data, ax=ax3)
	sbn.scatterplot(x="bp_rp", y="abs_magnitude", hue="bp_rp", palette="ch:s=1.0,r=0.1_r", hue_norm=(-1, 3), s=0.3, linewidth=0, legend=False, data=HR_data, ax=ax4)
	sbn.scatterplot(x="bp_rp", y="abs_magnitude", hue="bp_rp", palette="ch:s=1.0,r=0.1_r", hue_norm=(-1, 3), s=0.3, linewidth=0, legend=False, data=HR_data, ax=ax5)
	sbn.scatterplot(x="bp_rp", y="abs_magnitude", hue="bp_rp", palette="ch:s=1.0,r=0.1_r", hue_norm=(-1, 3), s=0.3, linewidth=0, legend=False, data=HR_data, ax=ax6)
	ax1.set_xlim([-1, 5])
	ax1.set_ylim([-5, 16])
	ax2.set_xlim([-1, 5])
	ax2.set_ylim([-5, 16])
	ax3.set_xlim([-1, 5])
	ax3.set_ylim([-5, 16])
	ax4.set_xlim([-1, 5])
	ax4.set_ylim([-5, 16])
	ax5.set_xlim([-1, 5])
	ax5.set_ylim([-5, 16])
	ax6.set_xlim([-1, 5])
	ax6.set_ylim([-5, 16])

	ax1.invert_yaxis()
	ax2.invert_yaxis()
	ax3.invert_yaxis()
	ax4.invert_yaxis()
	ax5.invert_yaxis()
	ax6.invert_yaxis()

#	if stars_df:
	sbn.scatterplot(x="bp_rp", y="abs_magnitude", color='k',
                alpha=0.75, s=9.0, linewidth=1.0, legend=False, data=rrab_df, ax=ax1)
	sbn.scatterplot(x="bp_rp", y="abs_magnitude", color='k',
                alpha=0.75, s=9.0, linewidth=1.0, legend=False, data=ea_df, ax=ax2)
	sbn.scatterplot(x="bp_rp", y="abs_magnitude", color='k',
                alpha=0.75, s=9.0, linewidth=1.0, legend=False, data=eb_df, ax=ax3)
	sbn.scatterplot(x="bp_rp", y="abs_magnitude", color='k',
                alpha=0.75, s=9.0, linewidth=1.0, legend=False, data=gdor_df, ax=ax4)
	sbn.scatterplot(x="bp_rp", y="abs_magnitude", color='k',
                alpha=0.75, s=9.0, linewidth=1.0, legend=False, data=dscut_df, ax=ax5)
	sbn.scatterplot(x="bp_rp", y="abs_magnitude", color='k',
                alpha=0.75, s=9.0, linewidth=1.0, legend=False, data=othper_df, ax=ax6)

	ax1.set_xlabel('')
	ax1.set_ylabel('')
	ax2.set_xlabel('')
	ax2.set_ylabel('')
	ax3.set_xlabel('')
	ax3.set_ylabel('')
	ax4.set_xlabel('')
	ax4.set_ylabel('')
	ax5.set_xlabel('')
	ax5.set_ylabel('')
	ax6.set_xlabel('')
	ax6.set_ylabel('')
	
	fig.add_subplot(111, frame_on=False)
	plt.tick_params(labelcolor="none", bottom=False, left=False)

	plt.xlabel('B-R Colour Index')
	plt.ylabel("Absolute Magnitude")
#	ax.set_xlabel('B-R Colour Index')
#	ax.set_ylabel('Absolute Magnitude')
	plt.tight_layout()
#	plt.show()
	plt.savefig('{}/final_report_images/HR_six.png'.format(project_dir), format='png')
	plt.savefig('{}/final_report_images/HR_six.pdf'.format(project_dir), format='pdf')
	plt.close()

def numbers_by_threshold(stars_df):
	rrab_df = stars_df[stars_df['Class'] == 'RRab']
	ea_df = stars_df[stars_df['Class'] == 'EA']
	eb_df = stars_df[stars_df['Class'] == 'EB']
	dscut_df = stars_df[stars_df['Class'] == 'DSCUT']
	gdor_df = stars_df[stars_df['Class'] == 'GDOR']
	othper_df = stars_df[stars_df['Class'] == 'OTHPER']
	noise_df = stars_df[stars_df['Class'] == 'Noise']
	print('Number of RRab Stars Classified: ', len(rrab_df))
	print('Number of EA Stars Classified: ', len(ea_df))
	print('Number of EB Stars Classified: ', len(eb_df))
	print('Number of DSCUT Stars Classified: ', len(dscut_df))
	print('Number of GDOR Stars Classified: ', len(gdor_df))
	print('Number of OTHPER Stars Classified: ', len(othper_df))
	print('Number of Noise Stars Classified: ', len(noise_df))
	print('========================\n')
	for campaign in [5, 6, 7, 8, 10]:
		print('\nTotal Number Classified in Campaign     {}: {}'.format(campaign, len(stars_df[stars_df['Campaign'] == campaign])))
		print('Number of RRab Classified in Campaign   {}: {}'.format(campaign, len(rrab_df[rrab_df['Campaign'] == campaign])))
		print('Number of EA Classified in Campaign     {}: {}'.format(campaign, len(ea_df[ea_df['Campaign'] == campaign])))
		print('Number of EB Classified in Campaign     {}: {}'.format(campaign, len(eb_df[eb_df['Campaign'] == campaign])))
		print('Number of DSCUT Classified in Campaign  {}: {}'.format(campaign, len(dscut_df[dscut_df['Campaign'] == campaign])))
		print('Number of GDOR Classified in Campaign   {}: {}'.format(campaign, len(gdor_df[gdor_df['Campaign'] == campaign])))
		print('Number of OTHPER Classified in Campaign {}: {}'.format(campaign, len(othper_df[othper_df['Campaign'] == campaign])))
		print('Number of Noise Classified in Campaign  {}: {}'.format(campaign, len(noise_df[noise_df['Campaign'] == campaign])))

	print('Prob\tRRab\tEA\tEB\tDSCUT\tGDOR\tOTHPER\tNoise')
	for prob in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
		rrab_val = len(rrab_df[rrab_df['RRab'] >= prob])
		ea_val = len(ea_df[ea_df['EA'] >= prob])
		eb_val = len(eb_df[eb_df['EB'] >= prob])
		dscut_val = len(dscut_df[dscut_df['DSCUT'] >= prob])
		gdor_val = len(gdor_df[gdor_df['GDOR'] >= prob])
		othper_val = len(othper_df[othper_df['OTHPER'] >= prob])
		noise_val = len(noise_df[noise_df['Noise'] >= prob])
		print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(prob, rrab_val, ea_val, eb_val, dscut_val, gdor_val, othper_val, noise_val))
	return

def training_set_composition(train_df):
	print(train_df)
	rrab_df = train_df[train_df['Class'] == '  RRab']
	ea_df = train_df[train_df['Class'] == '    EA']
	eb_df = train_df[train_df['Class'] == '    EB']
	dscut_df = train_df[train_df['Class'] == ' DSCUT']
	gdor_df = train_df[train_df['Class'] == '  GDOR']
	othper_df = train_df[train_df['Class'] == 'OTHPER']
	noise_df = train_df[train_df['Class'] == ' Noise']
	print(train_df)
	print('rrab', len(rrab_df))
	print('EA', len(ea_df))
	print('EB', len(eb_df))
	print('DSC', len(dscut_df))
	print('Gd', len(gdor_df))
	print('Oper', len(othper_df))
	print('Noise', len(noise_df))

	print('c1', len(train_df[train_df['Campaign']==1]))
	print('c2', len(train_df[train_df['Campaign']==2]))
	print('c3', len(train_df[train_df['Campaign']==3]))
	print('c4', len(train_df[train_df['Campaign']==4]))
	return

def plot_CM():

	plt.subplots(figsize=(15,12))
	cmlabels = ['DSCUT', 'EA', 'EB', 'GDOR', 'Noise', 'OTHPER', 'RRab']

	data = [[0.957, 0.,    0.031, 0.006, 0.006, 0.,    0.   ],
	        [0.,    0.898, 0.079, 0.,    0.015, 0.008, 0.   ],
	        [0.013, 0.063, 0.868, 0.,    0.,    0.056, 0.   ],
	        [0.032, 0.,    0.04,  0.798, 0.008, 0.122, 0.   ],
	        [0.008, 0.004, 0.,    0.008, 0.808, 0.172, 0.,   ],
	        [0.002, 0.,    0.008, 0.012, 0.26,  0.718, 0.   ],
	        [0.,    0.,    0.,    0.,    0.,    0.,    1.]]   

	sbn.heatmap(data, annot=True, linewidths=.75, cmap="GnBu",
	            xticklabels=cmlabels, yticklabels=cmlabels, fmt=".3f",
	            center=0.5, vmin=0, vmax=1, cbar_kws={'label': 'Proportion of Correct Classifications'})
	plt.yticks(rotation=0)
	plt.xlabel("Predicted Classes")
	plt.ylabel("True Classes")
	plt.tight_layout()
	plt.savefig('{}/final_report_images/CM.png'.format(project_dir), format='png')
	plt.savefig('{}/final_report_images/CM_transparent_2.pdf'.format(project_dir), format='pdf', transparent=True)
	plt.close()
	


def plot_fimp(test_df):
	cm_data = [[0.964, 0.0, 0.03, 0.006, 0.0,    0.0,    0.0 ],
	           [0., 0.907, 0.077, 0.008, 0., 0.008, 0.0],
	           [0.013, 0.063, 0.873, 0.006, 0.,    0.044, 0.   ],
	           [0.031, 0.,    0.031, 0.89,  0.,    0.047, 0.   ],
	           [0.003, 0.003, 0.003, 0.,    0.829, 0.16,  0.003],
	           [0.,    0.,    0.017, 0.02,  0.309, 0.654, 0.   ],
	           [0.,    0.,    0.,    0.,    0.,    0.,    1.   ]]

#	fimps = [0.154, 0.089, 0.02,  0.018, 0.013, 0.014, 0.012, 0.038, 0.041, 0.007, 0.058, 0.055,
#	         0.088, 0.057, 0.053, 0.04,  0.016, 0.069, 0.056, 0.036, 0.042, 0.025]
#
#	fimp_devs = [0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.0, 0.001, 0.001, 0.0,   
#	             0.002, 0.003, 0.003, 0.001, 0.002, 0.002, 0.001, 0.003, 0.003, 0.002, 0.002, 0.001]


	# XXX - RF 2D
	fimps = [0.153, 0.085, 0.021, 0.021, 0.013, 0.014, 0.012, 0.042, 0.039, 0.007, 0.053, 0.056,
	         0.086, 0.061, 0.052, 0.04, 0.018, 0.067, 0.059, 0.034, 0.04, 0.027]

	fimp_devs = [0.002, 0.004, 0.001, 0.001, 0.00, 0.001, 0.001, 0.002, 0.002, 0.00, 0.002, 0.003,
	             0.003, 0.002, 0.003, 0.002, 0.001, 0.003, 0.003, 0.002, 0.002, 0.001]

	# XXX - GB 1D
#	fimps = [0.186, 0.099, 0.021, 0.02, 0.011, 0.018, 0.011, 0.055, 0.051, 0.003, 0.063, 0.073,
#	         0.058, 0.039, 0.038, 0.082, 0.027, 0.088, 0.06]

#	fimp_devs = [0.004, 0.002, 0.002, 0.002, 0.002, 0.001, 0.002, 0.005, 0.002, 0.001, 0.002, 0.003,
#	             0.003, 0.007, 0.004, 0.005, 0.003, 0.013, 0.002]



	#sbn.set(style="whitegrid")

	#sbn.set_context("paper")
	test_df = test_df.drop('DJA_Class', axis=1).drop('class', axis=1).drop('probability', axis=1)
	features = test_df.columns[1:].to_numpy()
	print(features)
	fimp_df = pd.DataFrame(columns=['Feature', 'Score', 'Dev'])
	fimp_df['Feature'] = features
	fimp_df['Score'] = fimps
	arr = [fimps[i] + fimp_devs[i] for i in range(len(fimps))]
	fimp_df['Dev'] = fimp_devs
	print(fimp_df)
	fimp_df = fimp_df.sort_values(by='Score', ascending=False)
	print(fimp_df)

	fimp_df = fimp_df.head(5)


#	f, ax = plt.subplots(figsize=(6, 8))
	f, ax = plt.subplots(figsize=(8, 4))

	sbn.set_color_codes("muted")
	sbn.barplot(x='Score', y='Feature', data=fimp_df, color='b', label='Average', palette="GnBu_r", xerr=fimp_df['Dev'], error_kw={'elinewidth':1.5})
#	sbn.barplot(x='Score', y='Feature', data=fimp_df, color='b', label='Average', palette="YlOrBr_r", xerr=fimp_df['Dev'], error_kw={'elinewidth':1.5})
#	ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
	sbn.despine()

#	ax.set(xlim=(0, 0.195), ylabel="",
 #     xlabel="Relative GB Importance")
	ax.set(xlim=(0, 0.17), ylabel="",
       xlabel="Relative RF Importance")

	plt.tight_layout()
#	plt.show()

#	plt.savefig('{}/final_report_images/fimp_GB.png'.format(project_dir), format='png')
	plt.savefig('{}/final_report_images/fimp_RF_transparent_2.pdf'.format(project_dir), format='pdf', transparent=True)
	plt.close()


# 4 Way plot
def phasefold_examples():
	rrab_epic_6 = 212447818
	ea_epic_7 = 215343128
	eb_epic_5 = 211394059
	dscut_epic_7 = 219000406
	rrab_

def get_stars_and_gaia_composition():
	gaia_all = get_all_GAIA_data()
	print(gaia_all)
	total_campaign_stars = {0:7915, 1:21703, 2:16719, 3:17049, 4:17324, 5:25978, 6:47634, 7:15157, 8:30002, 9:6992, 10:83338, 11:65453, 12:46185, 13:26288, 14:39171, 15:35196, 16:35702, 17:46409, 18:37146, 19:44374}
	for campaign in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]:
		campaign_stars_file = open('{}/c{}_stars.txt'.format(project_dir, campaign), 'r')
		campaign_stars = campaign_stars_file.read().splitlines()
		campaign_stars_file.close()
		total_targets = total_campaign_stars[campaign]
		total_stars = len(campaign_stars)
		campaign_stars = [int(_) for _ in campaign_stars]
	#	print(campaign_stars)
		gaia_campaign = gaia_all[(gaia_all['epic_number'].isin(campaign_stars)) & (gaia_all['campaign_num'] == campaign)]
#		print(len(gaia_campaign))
#		print(len(set(gaia_campaign['epic_number'])))
		
#		print(len(campaign_stars))
	#	print(gaia_campaign)
		with_r_est = len(gaia_campaign['r_est'].dropna())
		with_bp_rp = len(gaia_campaign['bp_rp'].dropna())
		print("{} & {} & {} & {}".format(campaign, total_targets, total_stars, with_r_est))

def training_full_HR_diagram(gaia_train, gaia_all):

	fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 4), gridspec_kw={'wspace':0.05})

	sbn.scatterplot(x="bp_rp", y="abs_magnitude",
	               hue="bp_rp",
	               palette="ch:s=1.0,r=0.1_r",
	               hue_norm=(-1, 3),
	               s=1.3,
	               linewidth=0,
	               legend=False,
	               data=gaia_train, ax = ax1)
	sbn.scatterplot(x="bp_rp", y="abs_magnitude",
	               hue="bp_rp",
	               palette="ch:s=1.0,r=0.1_r",
	               hue_norm=(-1, 3),
	               s=0.3,
	               linewidth=0,
	               legend=False,
	               data=gaia_all, ax = ax2)
	ax1.set_xlim([-1, 5])
	ax1.set_ylim([-5, 16])
	ax2.set_xlim([-1, 5])
	ax2.set_ylim([-5, 16])
	ax1.invert_yaxis()
	ax2.invert_yaxis()
	ax1.set_xlabel('')
	ax1.set_ylabel('')
	ax2.set_xlabel('')
	ax2.set_ylabel('')

	fig.add_subplot(111, frame_on=False)
	plt.tick_params(labelcolor="none", bottom=False, left=False)

	plt.xlabel('B-R Colour Index')
	plt.ylabel("Absolute Magnitude")

	plt.subplots_adjust()
	plt.tight_layout()
#	plt.show()
	plt.savefig('{}/final_report_images/HR_train_full_side_by_side.png'.format(project_dir), format='png')
	plt.savefig('{}/final_report_images/HR_train_full_side_by_side.pdf'.format(project_dir), format='pdf')
	plt.close()


def results():
	total_df = pd.read_csv('{}/src/models/{}_{}/unknown_predictions_{}.csv'.format(project_dir, model_name, training_set, model_number))
	c5_stars_file = open('{}/known/c5_stars.txt'.format(project_dir), 'r')
	c5_stars = c5_stars_file.read().splitlines()
	c5_stars_file.close()
	c6_stars_file = open('{}/known/c6_stars.txt'.format(project_dir), 'r')
	c6_stars = c6_stars_file.read().splitlines()
	c6_stars_file.close()
	c7_stars_file = open('{}/known/c7_stars.txt'.format(project_dir), 'r')
	c7_stars = c7_stars_file.read().splitlines()
	c7_stars_file.close()
	c8_stars_file = open('{}/known/c8_stars.txt'.format(project_dir), 'r')
	c8_stars = c8_stars_file.read().splitlines()
	c8_stars_file.close()
	c10_stars_file = open('{}/known/c10_stars.txt'.format(project_dir), 'r')
	c10_stars = c10_stars_file.read().splitlines()
	c10_stars_file.close()
	stars = np.concatenate((c5_stars, c6_stars, c7_stars, c8_stars, c10_stars))
	periods = collate_periods()
	stars_df = total_df[total_df['epic_number'].isin(stars)]
	classes = ['RRab', 'EA', 'EB', 'GDOR', 'DSCUT']
	test_df = pd.read_csv('{}/src/models/{}_{}/test.csv'.format(project_dir, model_name, training_set))

#	rrab_candidates = stars_df[stars_df['Class'] == 'RRab'] 
#	ea_candidates = stars_df[stars_df['Class'] == 'EA'] 
#	eb_candidates = stars_df[stars_df['Class'] == 'EB'] 
#	gdor_candidates = stars_df[stars_df['Class'] == 'GDOR'] 
#	dscut_candidates = stars_df[stars_df['Class'] == 'DSCUT'] 
#	noise_candidates = stars_df[stars_df['Class'] == 'Noise']
#	othper_candidates = stars_df[stars_df['Class'] == 'OTHPER']


	# Probability Distributions
	# Period Distributions
	# HR Diagram
	# Plot in decreasing probability order - give thresholds
	# Give number of classified items per band of probabilty threshold

#	plot_unknown(classes, stars_df, periods)

#	probability_thresholds(stars_df, classes)

#	feature_distribution(stars_df, classes, test_df)

#	HR_data = get_GAIA_data()
#	HR_diagram(HR_data, stars_df, test_df)

	# XXX - COME BACK TO IF HAVE TIME.
#	feature_dist_2(stars_df, classes, test_df)

	# TODO Get Classification statistics by threshold.
	#numbers_by_threshold(stars_df)

#	train_df = pd.read_csv('{}/training_sets/k2sc/c1-4_{}.csv'.format(project_dir, training_set))
#	training_set_composition(train_df)
#	gaia_train = get_training_GAIA_data(train_df)
#	gaia_all = get_all_GAIA_data()
#	HR_diagram(gaia_all)

#	test_df = pd.read_csv('{}/src/models/som_and_rf_delta/train.csv'.format(project_dir))
#	ax3.set_xlabel("Time (BJD - 2454833)")

#	plot_fimp(test_df)

#	get_stars_and_gaia_composition()
#	training_feature_distribution()

#	average_train_probabilities()
#	training_full_HR_diagram(gaia_train, gaia_all)

	plot_CM()




def main():
	results()
#	get_GAIA_data()

if __name__ == '__main__':
	main()

