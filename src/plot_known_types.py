from kepler_data_utilities import *

star_types = ["RRab", "DSCUT", "EA", "EB", "GDOR", "Noise"]
campaign_num = 3

for type in star_types:
	plot_ids = []
	fig, axs = plt.subplots(3, 3)
	with open("../known/{}_{}.txt".format(campaign_num, type)) as file:
		plot_ids = [next(file).strip() for x in range(9)]
	for i, ax in enumerate(axs.flat):
		hdul = get_hdul(False, plot_ids[i])
		times, flux = get_lightcurve(hdul, "PDC")
		ax.plot(times[:200], flux[:200]/1000, linewidth=0.5)
	fig.suptitle("Known {} stars from c-0{} (Raw K2SC PDC Data ~ First 100 hours)"\
	             .format(type, campaign_num))
	plt.subplots_adjust(wspace=0.5, hspace=0.5)
	fig.text(0.5, 0.02, 'Time (Days)', ha='center')
	fig.text(0.02, 0.5, 'Flux / (10^3)', va='center', rotation='vertical')
	plt.savefig("known_{}_c0{}_plot.png".format(type, campaign_num))
	plt.close()
