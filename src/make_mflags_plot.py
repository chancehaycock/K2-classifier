from kepler_data_utilities import *

# Plots to determine which flags are causing problems in the periodogram.

def make_flags_plot(campaign_num, plot=False):

	flag_ids = {0 : "K2 Quality", 1 : "Transit", 2 : "Flare", 3 : "Upwards Outlier",
	            4 : "Downwards Outlier", 5 : "Non-Finite Flux", 6 : "Periodic Mask"}
	colours = ['r', '', '', 'g', 'c', 'y', 'm']

	star_types = ["RRab", "DSCUT", "EA", "EB", "GDOR"]
	hist_dict = {}
	for star_type in star_types:
		for flag in flag_ids.values():
			hist_dict['{} {}'.format(star_type, flag)] = 0

	# Test Epic_number
	epic_num = 206143957 

	data_file = '{}/tables/campaign_{}_master_table.csv'.format(px402_dir, campaign_num)
	df = pd.read_csv(data_file)

	# TODO - DONT LIKE THIS! Use String! USE PDC lightcurves
	lc_type_indx = 1


	down_outlier_array = []

	for i in range(len(df['epic_number'])):
		epic_num = int(df.iloc[i]['epic_number'])
		star_class = str(df.iloc[i]['Class'])
		star_class = star_class.split()[0]

		if star_class in star_types:
			hdul = get_hdul(use_remote, epic_num, campaign_num)

			flux = hdul[lc_type_indx].data['flux']
			trend_t = hdul[lc_type_indx].data['trtime']
			times = hdul[lc_type_indx].data['time']
			mflags = hdul[lc_type_indx].data['mflags']

			# Note that this is the detrended data as described above.
			flux_c = flux + trend_t - np.median(trend_t)

			flags_off_del_array = np.nonzero(mflags)
			flags_off_flux = np.delete(flux_c, flags_off_del_array)
			flags_off_times = np.delete(times, flags_off_del_array)

			down_outlier_count = 0
			for target_flag in flag_ids:
				flag_del_array = []
				count = 0
				for i, flag in enumerate(mflags):
					if 2 ** target_flag <= flag <= (2 ** (target_flag + 1)) - 1:
						count += 1
					else:
						flag_del_array.append(i)

					# Downward Outlier Check - really not an efficient way of 
					# doing this, but it works.
					if 2 ** 4 <= flag <= (2 ** (4 + 1)) - 1 and (star_class == "EA" or star_class == "EB"):
						down_outlier_count += 1

				target_flag_flux = np.delete(flux_c, flag_del_array)
				target_flag_times = np.delete(times, flag_del_array)

				hist_dict["{} {}".format(star_class, flag_ids[target_flag])] += count
				

				if plot:
					plt.scatter(target_flag_times, target_flag_flux, s=5.0,
					            c=colours[target_flag],
					            label='{}'.format(flag_ids[target_flag]))
			if (star_class == "EA" or star_class == "EB"):
				down_outlier_array.append(down_outlier_count)

			if plot:
				plt.scatter(flags_off_times, flags_off_flux, s=1.0)
				plt.legend()
				plot_dir = "{}/plots/flags/".format(px402_dir)
				plt.savefig("{}/{}_flag_plot_{}_c{}.png"\
				            .format(plot_dir, star_class, epic_num, campaign_num))
				plt.close()

	print(down_outlier_array)
	print(len(down_outlier_array))
	print("Mean: ", np.mean(down_outlier_array))
	print("Median: ", np.median(down_outlier_array))
	print("Stddev: ", np.std(down_outlier_array))
	plt.hist(down_outlier_array, bins=100)
	plt.xlabel("Number of Downward Outliers Per Star") 
	plt.ylabel("Count")
	plt.title("Downward Outlier Distribution for Campaign 3 Eclipsing Binaries")
	plt.show()

# {'RRab K2 Quality': 5207, 'RRab Transit': 0, 'RRab Flare': 0, 'RRab Upwards Outlier': 2808,
# 'RRab Downwards Outlier': 2400, 'RRab Non-Finite Flux': 11519, 'RRab Periodic Mask': 0,
# 'DSCUT K2 Quality': 13489, 'DSCUT Transit': 0, 'DSCUT Flare': 0, 'DSCUT Upwards Outlier': 1077,
# 'DSCUT Downwards Outlier': 156, 'DSCUT Non-Finite Flux': 17497, 'DSCUT Periodic Mask': 0,
# 'EA K2 Quality': 10229, 'EA Transit': 0, 'EA Flare': 0, 'EA Upwards Outlier': 3086,
# 'EA Downwards Outlier': 3068, 'EA Non-Finite Flux': 16307, 'EA Periodic Mask': 0,
# 'EB K2 Quality': 8060, 'EB Transit': 0, 'EB Flare': 0, 'EB Upwards Outlier': 1574,
# 'EB Downwards Outlier': 4782, 'EB Non-Finite Flux': 10924, 'EB Periodic Mask': 0,
# 'GDOR K2 Quality': 9575, 'GDOR Transit': 0, 'GDOR Flare': 0, 'GDOR Upwards Outlier': 1518,
# 'GDOR Downwards Outlier': 1040, 'GDOR Non-Finite Flux': 13859, 'GDOR Periodic Mask': 0}


def plot_flag_hist(input_dict, plot_type):

	plt.bar(input_dict.keys(), input_dict.values(), color='g')
	plt.xticks(rotation=70)
	plt.ylabel("Flag Count")
	plt.title("Campaign 3 Flag Stats - {}".format(plot_type))
	plt.tight_layout()
	plot_dir = "{}/plots/flags/".format(px402_dir)
	plt.savefig("{}/STATS_{}_flag_plot_c3.png"\
	            .format(plot_dir, plot_type))
	plt.close()


def main():
	make_flags_plot(3)

	RRab_dict = {'RRab K2 Quality': 5207, 'RRab Transit': 0, 'RRab Flare': 0, 'RRab Upwards Outlier': 2808,
	             'RRab Downwards Outlier': 2400, 'RRab Non-Finite Flux': 11519, 'RRab Periodic Mask': 0}
	DSCUT_dict = {'DSCUT K2 Quality': 13489, 'DSCUT Transit': 0, 'DSCUT Flare': 0, 'DSCUT Upwards Outlier': 1077,
	              'DSCUT Downwards Outlier': 156, 'DSCUT Non-Finite Flux': 17497, 'DSCUT Periodic Mask': 0}
	EA_dict = {'EA K2 Quality': 10229, 'EA Transit': 0, 'EA Flare': 0, 'EA Upwards Outlier': 3086,
	           'EA Downwards Outlier': 3068, 'EA Non-Finite Flux': 16307, 'EA Periodic Mask': 0}
	EB_dict = {'EB K2 Quality': 8060, 'EB Transit': 0, 'EB Flare': 0, 'EB Upwards Outlier': 1574,
	           'EB Downwards Outlier': 4782, 'EB Non-Finite Flux': 10924, 'EB Periodic Mask': 0}
	GDOR_dict = {'GDOR K2 Quality': 9575, 'GDOR Transit': 0, 'GDOR Flare': 0, 'GDOR Upwards Outlier': 1518,
	             'GDOR Downwards Outlier': 1040, 'GDOR Non-Finite Flux': 13859, 'GDOR Periodic Mask': 0}

#	plot_flag_hist(RRab_dict, "RRab")
#	plot_flag_hist(DSCUT_dict, "DSCUT")
#	plot_flag_hist(EA_dict, "EA")
#	plot_flag_hist(EB_dict, "EB")
#	plot_flag_hist(GDOR_dict, "GDOR")

	K2_quality_dict = {'RRab K2 Quality': 5207, 'DSCUT K2 Quality': 13489,
	                     'EA K2 Quality': 10229,'EB K2 Quality': 8060,'GDOR K2 Quality': 9575}
	up_outlier_dict = {'RRab Upwards Outlier': 2808,'DSCUT Upwards Outlier': 1077,
	                   'EA Upwards Outlier': 3086,'EB Upwards Outlier': 1574,
	                   'GDOR Upwards Outlier': 1518}
	down_outlier_dict = {'RRab Downwards Outlier': 2400,'DSCUT Downwards Outlier': 156,
	                     'EA Downwards Outlier': 3068,'EB Downwards Outlier': 4782,
	                     'GDOR Downwards Outlier': 1040}
	inf_flux_dict = {'RRab Non-Finite Flux': 11519,'DSCUT Non-Finite Flux': 17497,
	                 'EA Non-Finite Flux': 16307,'EB Non-Finite Flux': 10924,
	                 'GDOR Non-Finite Flux': 13859}

#	plot_flag_hist(K2_quality_dict, "K2_Quality")
#	plot_flag_hist(up_outlier_dict, "Upward_Outliers")
#	plot_flag_hist(down_outlier_dict, "Downward_Outliers")
#	plot_flag_hist(inf_flux_dict, "Non-Finite_Flux")

if __name__ == "__main__":
	main()
