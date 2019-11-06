from kepler_data_utilities import *

def plot(star_type, feature, bins=100):
	df = pd.read_csv('{}/cross_match_data/gaia/campaign_3_mixed_table.csv'\
	                 .format(px402_dir))
	df = df.dropna()
	df = df[df['Class'] == star_type]
	df = df[feature]
	array = df.to_numpy()
	plt.hist(array, bins=bins)
	plt.title("Type: {}, Quantity: {}".format(star_type.strip(), feature))
	plt.xlabel("{}".format(feature))
	plt.ylabel("Count")
	plt.show()
	print("Plot Complete.")

def main():
	#plot("  RRab", "r_est")
	plot(" Noise", "r_est", 200)

if __name__ == "__main__":
	main()
