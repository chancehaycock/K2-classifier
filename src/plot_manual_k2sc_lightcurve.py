from kepler_data_utilities import *
import os 

file_dir = '/Users/chancehaycock/dev/machine_learning/px402/lightcurve_data/k2sc/campaign_1-2_manual'

for filee in os.listdir(file_dir):
	epic = filee[5:14]
	print(epic)
	hdul=fits.open('{}/{}'.format(file_dir, filee))

	raw_flux = hdul[1].data['flux']
	trend_t = hdul[1].data['trtime']
	times = hdul[1].data['time']
	mflags = hdul[1].data['mflags']
	err = hdul[1].data['error']

	flux = raw_flux + trend_t - np.median(trend_t)

	use_times, use_flux = get_lightcurve(hdul)

	fig, (ax1, ax2) = plt.subplots(2, 1)
	ax1.scatter(times, raw_flux, s=0.5, label='pdc')
	ax1.scatter(times, flux, s=0.5, label='k2sc', c='orange')
	fig.suptitle("EPIC {}".format(epic))
	ax1.legend()
	ax2.scatter(use_times, use_flux, s=0.5, label='k2sc - processed outliers', c='orange')
	ax2.legend()

	plt.savefig('{}/plots/k2sc_vs_pdc_c1-2_attempt2/{}_k2sc_vs_pdc.png'.format(project_dir, epic))
	plt.close()
