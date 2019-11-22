# This script searches through armstrong_0_to_4.csv, and plots histograms
# of the classificiation probabilities for each star type. This should give us
# some sort of handle on what sort of threshold each star type requires to be
# accurately classified as that type. 

from kepler_data_utilities import *

def plot_probability_histograms(campaign_num):
	df = pd.read_csv("{}/known/armstrong_0_to_4.csv".format(px402_dir))
	df = df[df["Campaign"] == campaign_num]

	star_types = ["  RRab", "    EA", "    EB", " DSCUT", "  GDOR"]

	#print(df[df["Class"] == "  RRab"])

	for star_type in star_types:
		probabilities = []
		class_df = df[df["Class"] == star_type]
		for i in range(len(class_df["Class"])):
			probabilities.append(class_df.iloc[i][star_type])
		plt.hist(probabilities, bins=10)
		plt.title("Classification Probabilities for {}".format(star_type.split()[0]))
		plt.xlabel("Probability")
		plt.ylabel("Count")
		plt.xlim((0, 1))
		plt.savefig("{}/plots/classification_probabilities_plots/{}_c{}.png"\
		            .format(px402_dir, star_type.split()[0], campaign_num))
		plt.close()


# ===================
#    Main Function
# ===================

def main():
	plot_probability_histograms(3)

if __name__ == "__main__":
	main()
