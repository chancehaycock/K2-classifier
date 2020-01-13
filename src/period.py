# An optimistic attempt to create a script which automatically finds periods
# of the inputted lightcurves, as opposed to passing them on to SJH. This is
# mainly so that I personally have a better understanding of the process.

# Once complete, I will compare my result with periods calculated by SJH and 
# supervisor David Armstrong.

from kepler_data_utilities import *

def get_lightcurve_periods(epic_num):
	ev_hdul = get_hdul(epic_num, 3, detrending="everest")
	ev_time, ev_flux = get_lightcurve(ev_hdul, process_outliers=False, detrending='everest', include_errors=False)
	k2sc_hdul = get_hdul(epic_num, 3, detrending="k2sc")
	k2sc_time, k2sc_flux = get_lightcurve(k2sc_hdul, process_outliers=False, detrending='k2sc', include_errors=False)
#	plt.errorbar(time, flux, yerr=err, ecolor='r')
#	plt.show()
#	freq, power = LombScargle(time, flux, err).autopower(nyquist_factor=2)
#	plt.plot(freq, power)
	plt.scatter(ev_time, ev_flux, label='everest', s=0.5)
	plt.scatter(k2sc_time, k2sc_flux, label='k2sc', s=0.5)
	plt.legend()
	plt.show()


def main():
	test_epic = 205945953
	ev_test_epic = 205898160
#	ev_test_epic = 205992558
	ev_test_epic = 205915657
	get_lightcurve_periods(ev_test_epic)

if __name__ == "__main__":
	main()
