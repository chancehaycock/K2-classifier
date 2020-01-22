from kepler_data_utilities import *

#hdul=fits.open('pipelines/k2sc/example_run/EPIC_205938750_mast.fits')
hdul=fits.open('pipelines/k2sc/example_run/EPIC_201129544_mast.fits')
lc_type_index='PDC'
hdul2=fits.open('pipelines/k2sc/example_run/ktwo201129544-c01_llc.fits')

time = hdul2[1].data['TIME']
sap_fluxes = hdul2[1].data['SAP_FLUX']
pdcsap_fluxes = hdul2[1].data['PDCSAP_FLUX']
plt.scatter(time, sap_fluxes, s=0.5, label='sap')
plt.scatter(time, pdcsap_fluxes, s=0.5, label='pdc')
plt.legend()
plt.show()
plt.close()

raw_flux = hdul[1].data['flux']
trend_t = hdul[1].data['trtime']
times = hdul[1].data['time']
mflags = hdul[1].data['mflags']
err = hdul[1].data['error']

flux = raw_flux + trend_t - np.median(trend_t)

plt.scatter(times, raw_flux, s=0.5, label='sap_raw')
plt.scatter(times, flux, s=0.5, label='sap_k2sc')
plt.legend()
plt.show()
