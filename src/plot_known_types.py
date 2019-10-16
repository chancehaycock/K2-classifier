from kepler_data_utilities import *

star_types = ["RRab", "DSCUT", "EA", "EB", "GDOR", "Noise"]
campaign_num = 4

for type in star_types:
	plot_ids = []
	fig, axs = plt.subplots(3, 3)
	with open("../known/{}_{}.txt".format(campaign_num, type)) as file:
		plot_ids = [next(file).strip() for x in range(9)]
	for i, ax in enumerate(axs.flat):
		hdul = get_hdul(False, plot_ids[i])
		times, flux = get_lightcurve(hdul, "PDC")
		ax.plot(times[:200], flux[:200], linewidth=0.5)
	fig.suptitle("Known {} stars from Campaign {} (Raw K2SC PDC Data ~ First 100 hours)".format(type, campaign_num))
	plt.subplots_adjust(wspace=0.5, hspace=0.5)
	plt.savefig("known_{}_c0{}_plot.png".format(type, campaign_num))
	plt.close()
