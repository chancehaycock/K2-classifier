# A script to plot the correlation between our periods and the ones found in
# Armstrong et. al 2016. At the time of writing, our code explors up until 
# 20 days as a maximum. If the most significant period is above this threshold,
# our code maxes out at 20.

from kepler_data_utilities import *

def plot_period_correlation(detrending='k2sc'):
	dja_periods = pd.read_csv("{}/periods/armstrong.csv".format(project_dir))
	sjh_1 = pd.read_csv("{}/periods/{}/campaign_1.csv".format(project_dir, detrending))
	sjh_2 = pd.read_csv("{}/periods/{}/campaign_2.csv".format(project_dir, detrending))
	sjh_3 = pd.read_csv("{}/periods/{}/campaign_3.csv".format(project_dir, detrending))
	sjh_4 = pd.read_csv("{}/periods/{}/campaign_4.csv".format(project_dir, detrending))
	sjh_periods = sjh_1.append(sjh_2, ignore_index=True).append(sjh_3, ignore_index=True).append(sjh_4, ignore_index=True)

	training_set = pd.read_csv('{}/training_sets/{}/c1-4_alpha.csv'.format(project_dir, detrending))


	known_df = pd.read_csv("{}/known/armstrong_0_to_4.csv".format(project_dir))

#	print(dja_periods)
#	print(sjh_periods)
	print(known_df)


	rrab_points = []
	ea_points = []
	eb_points = []
	gdor_points = []
	dscut_points = []
	othper_points = []
	noise_points = []
	points = {'  RRab': rrab_points, '    EA': ea_points, '    EB': eb_points, '  GDOR': gdor_points,
	         ' DSCUT': dscut_points, 'OTHPER': othper_points, ' Noise': noise_points}

	classes = points.keys()
	print(classes)

	period_plot_df = pd.DataFrame(columns=['epic', 'DJA_period', 'SJH_period', 'class', 'probability'])
	for i, epic in enumerate(sjh_periods['epic_number']):
		sjh_period = sjh_periods[sjh_periods['epic_number'] == epic].iloc[0]['Period_1']
		dja_row = dja_periods[dja_periods['epic_number'] == epic]
		if dja_row.empty:
			print("Epic Not Found in Armstrong Table.")
			continue 
		dja_period = dja_row.iloc[0][' period']
		star_type = known_df[known_df['epic_number'] == epic].iloc[0]['Class']
		probability = known_df[known_df['epic_number'] == epic][classes].max(axis=1).iloc[0]
		#print(probability)
		if probability < 0.5:
			continue
		points[star_type].append([sjh_period, dja_period])

		add_array_df = pd.DataFrame([[epic, dja_period, sjh_period, star_type, probability]], columns=['epic', 'DJA_period', 'SJH_period', 'class', 'probability'])
#		add_array_df.iloc[0]['epic'] = epic
#		add_array_df.iloc[0]['DJA_period'] = dja_period
#		add_array_df.iloc[0]['SJH_period'] = sjh_period
#		add_array_df.iloc[0]['class'] = star_type
#		add_array_df.iloc[0]['probability'] = probability
		#print(add_array_df)
		period_plot_df = period_plot_df.append(add_array_df)
		
	print(period_plot_df)
	period_plot_df.to_csv('period_plot.csv')
	return

def plot_it():
	period_plot_df = pd.read_csv('period_plot.csv')
	print(period_plot_df)
	period_plot_df = period_plot_df[period_plot_df['class'] != ' Noise']
#	period_plot_df = period_plot_df[period_plot_df['class'] != 'OTHPER']
	markers = ['v', 'X', 'o', 'p', 'd', '^']
	flatui = ["#9b59b6", "#95a5a6", "#3498db", "#e74c3c", "#34495e"]#, "#2ecc71"]
	palette4 = sbn.color_palette(flatui)
	fig, ax = plt.subplots()
	sbn.scatterplot(x='SJH_period', y='DJA_period', palette=palette4, #style='class',
	                    #linewidth=0.75,
	                    linewidth=0.00,
	                    #hue='class',
	                    s=2.0, #sizes=[30 + 5, 15 + 5, 20 + 5, 25 + 5, 35 + 5, 30 + 5], size='class',
	                    alpha=1.0, color=flatui[4], data=period_plot_df, legend=False)#, markers=markers)

	ax.set_xlabel('Calculated Period / Days')
	ax.set_ylabel('Armstrong Period / Days')
#	print(points)
#	fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
#	ax1.scatter(*zip(*points['  RRab']), s=0.5, label="RRab")
#	ax1.set(xlim=(0, 2), ylim=(0, 2))
#	ax1.legend()


#	ax2.scatter(*zip(*points['    EA']), s=0.5, label="EA")
#	ax2.set(xlim=(0, 20), ylim=(0, 20))
#	ax2.legend()

#	ax3.scatter(*zip(*points['    EB']), s=0.5, label="EB")
#	ax3.set(xlim=(0, 5), ylim=(0, 5))
#	ax3.legend()

#	ax4.scatter(*zip(*points['  GDOR']), s=0.5, label="GDOR")
#	ax4.set(xlim=(0, 4), ylim=(0, 4))
#	ax4.legend()

#	ax5.scatter(*zip(*points[' DSCUT']), s=0.5, label="DSCUT")
#	ax5.set(xlim=(0, 0.2), ylim=(0, 0.2))
#	ax5.legend()

#	ax6.scatter(*zip(*points['OTHPER']), s=0.5, label="OTHPER")
#	ax6.set(xlim=(0, 20), ylim=(0, 20))
#	ax6.legend()

	plt.tight_layout()
#	plt.show()
	plt.savefig("{}/final_report_images/period_correlation.pdf".format(project_dir), format='pdf')
	plt.close()

#	plt.scatter(*zip(*points[' Noise']), s=0.5, label="Noise only")
#	plt.xlim(0, 20)
#	plt.ylim(0, 20)
#	plt.legend()
#	plt.savefig("{}/plots/alpha_period_correlation_noise.pdf".format(project_dir), format='pdf')
#	plt.close()




def main():
#	plot_period_correlation()
	plot_it()

if __name__ == "__main__":
	main()
