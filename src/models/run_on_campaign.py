# This script is just temporary to confirm the behaviour SJH experienving when
# running beta model on the whole of campaign 4 minus training data.

from models.model_utilities import *

def run_on_campaign(campaign_num, training):

	# These are currently set to the best parameteres found for the beta training set
	best_parameters = {'max_features': 4, 'min_samples_split': 3, 'n_estimators': 500}

	training_file = "{}/src/models/som_and_rf_{}/train.csv".format(project_dir, training)
	# Read in training file
	train_df = pd.read_csv(training_file)

	# Features that we are interested in testing
	# features = train_df.drop('class', axis=1).drop('probability', axis=1).columns[1:len(train_df.columns)-2]
	features = choose_features(train_df, include_som=True, include_period=True,
                               include_colour=True, include_absmag=True,
                               include_lc_stats=True, include_bin_lc_stats=True)
	print(features)
	file_name = "everything"

	# List of Epics
	training_epics = train_df['epic_number'].to_numpy()


	# Now import/construct test file
	master = pd.read_csv('{}/tables/k2sc/campaign_{}_master_table.csv'.format(project_dir, campaign_num))
	som = pd.read_csv('{}/som_statistics/k2sc/c1-4_{}/campaign_{}.csv'.format(project_dir, training, campaign_num))
	test_df = master.merge(som, how='left', on='epic_number') 
	bin_columns = make_bin_columns(64)
	test_df = test_df.drop(bin_columns, axis=1)

	# Remove members from the test set that are already present in the training set
	test_df = test_df[~test_df['epic_number'].isin(training_epics)]
	# Remove members that aren't classified by Armstrong et. al
	test_df = test_df[test_df['class'] != 'NOT CLASSIFIED']

	print("TRAINING SET")
#	train_df = train_df.dropna()
	print(train_df)
	print("TEST SET")
	test_df = test_df.dropna()
	print(test_df)

	# Plot confusion matrix
	clf = RCF(random_state=2, class_weight='balanced', n_estimators=300, max_features='auto')
	clf.fit(train_df[features], train_df['class'])
	test_predictions = clf.predict(test_df[features])
	CM = confusion_matrix(test_df['class'], test_predictions)
	cmlabels = np.sort(clf.classes_) 

	sumarr = CM.astype(np.float).sum(axis=1)
	normCM = [CM[i] / sumarr[i] for i in range(len(sumarr))] 
	sbn.heatmap(normCM, annot=True, linewidths=.75, cmap="YlGnBu",
	            xticklabels=cmlabels, yticklabels=cmlabels, fmt=".2f",
	            center=0.5, vmin=0, vmax=1)
	plt.xlabel("Predicted Classes")
	plt.ylabel("True Classes")
	plt.title("{}".format(file_name))
	plt.tight_layout()
	plt.savefig('{}/src/models/som_and_rf_{}/campaign_test_runs/{}_tested_on_campaign_{}_{}.png'.format(project_dir, training, training, campaign_num, file_name))
	plt.close()

	# Get feature importances
	fimp = clf.feature_importances_

	plt.bar(features, fimp)
	plt.xticks(rotation=90)
	plt.tight_layout()
	plt.savefig('{}/src/models/som_and_rf_{}/campaign_test_runs/{}_tested_on_campaign_{}_{}_fimp.png'.format(project_dir, training, training, campaign_num, file_name))
	plt.close()

	# Get probability estimates to write to csv file.
	this_model_probs = clf.predict_proba(test_df[features])
	this_model_max_probs = np.max(this_model_probs, axis=1)

	out_df = pd.DataFrame()
	out_df['epic_number'] = test_df['epic_number']
	out_df['this_model_class'] = test_predictions
	out_df['this_model_probs'] = this_model_max_probs
	out_df['DA_model_class'] = test_df['class']
	out_df['DA_model_probs'] = test_df['probability']
	out_df['period'] = test_df['Period_1']
	print(out_df)
	print("The above dataframe has been written to csv.")
	out_df.to_csv('{}/src/models/som_and_rf_{}/campaign_test_runs/campaign_{}_test_results.csv'.format(project_dir, training, campaign_num), index=False)
	return


# Used to compare the classifications made in Armstrong et. al. versus ours for
# a whole campaign. As is tands, we are getting horrendous behaviou, particularly with Gamme Dors.
def make_processing_plot(campaign_num, training):

#	csv_file = '{}/src/models/som_and_rf_{}/campaign_test_runs/campaign_{}_test_results.csv'.format(project_dir, training, campaign_num)
	csv_file = '{}/training_sets/k2sc/c1-4_point5.csv'.format(project_dir)
	df = pd.read_csv(csv_file)

	changed_epics_file = '{}/training_sets/k2sc/manual_changes_sjh.csv'.format(project_dir)
	changed_epics_df = pd.read_csv(changed_epics_file)
	changed_epics = changed_epics_df['epic_number'].to_numpy()
#	df = df[(df['Class'] == ' Noise') & (df['Class'] != 'OTHPER')]
	df = df[df['epic_number'].isin(changed_epics)]

	# IMport all periods and merge
	period_c1 = pd.read_csv('{}/periods/k2sc/campaign_1.csv'.format(project_dir))
	period_c2 = pd.read_csv('{}/periods/k2sc/campaign_2.csv'.format(project_dir))
	period_c3 = pd.read_csv('{}/periods/k2sc/campaign_3.csv'.format(project_dir))
	period_c4 = pd.read_csv('{}/periods/k2sc/campaign_4.csv'.format(project_dir))
	period_df = period_c1.append(period_c2, ignore_index=True).append(period_c3, ignore_index=True).append(period_c4, ignore_index=True)


	for i in range(len(df['epic_number'])):

#		epic_num = int(df.iloc[i]['epic_number'])
#		dja_class = df.iloc[i]['DA_model_class']
#		dja_prob = df.iloc[i]['DA_model_probs']
#		this_class = df.iloc[i]['this_model_class']
#		this_prob = df.iloc[i]['this_model_probs']
#		period = df.iloc[i]['period']

		epic_num = int(df.iloc[i]['epic_number'])
#		dja_class = df.iloc[i]['Class'].strip()
#		dja_prob = df.iloc[i]['Probability']
		campaign_num = df.iloc[i]['Campaign']
		period = period_df[period_df['epic_number'] == epic_num].iloc[0]['Period_1']
		dja_class = df.iloc[i]['Class'].strip()
		updated_class = changed_epics_df[changed_epics_df['epic_number'] == epic_num].iloc[0]['updated_class'].strip()

		n_bins=64

		# Processing of the lightcurve begins here
		hdul = get_hdul(epic_num, campaign_num)
		# By default chooses PDC
		times, flux = get_lightcurve(hdul)
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

	#	fig.suptitle("Period: %.3f DJA Prob: %.2f This Prob: %.2f" % (period, dja_prob, this_prob))
		fig.suptitle("DJA Class: {} Proposed Class {}".format(dja_class, updated_class))
		plot_dir = "{}/plots/everything_over_point5"\
		           .format(px402_dir)
		plt.tight_layout()
		plt.subplots_adjust(top=0.85)
		#plt.show()
		plt.savefig("{}/{}/{}_{}?.png"\
		.format(plot_dir, dja_class, epic_num, updated_class))
		plt.close()


def main():
	# Choose the training set to train the RF on.
	training = 'gamma'
	# Choose the campaign num to test on. Any members which are in the
	# above chosen training set, are removed.
	campaign_num = 4


	# Run this first to get confusion matrix and csv file of probability predicitions
#	run_on_campaign(campaign_num, training)

	# To plot the lightcurves, then run this...
	make_processing_plot(campaign_num, training)


if __name__ == "__main__":
	main()
