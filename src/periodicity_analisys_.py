import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import BoxLeastSquares

from astroquery.mast import Catalogs
from astroquery.gaia import Gaia


from astropy.io import fits
from transitleastsquares import (
    transitleastsquares,
    catalog_info,
    cleaned_array
)


class BLSP_model:
    def __init__(self, t, y_filt):
        self.t = t
        self.y_filt = y_filt

    def calculate_period(self):
        durations = np.linspace(0.05, 0.2, 20)
        model_bls = BoxLeastSquares(self.t, self.y_filt)
        results_bls = model_bls.autopower(durations, frequency_factor=10)
        period = results_bls.period[np.argmax(results_bls.power)]
        return period, results_bls

    def plot_periodogram(self, results_bls, period):
        plt.figure()
        ax = plt.gca()
        ax.axvline(period, alpha=0.4, lw=3)
        for n in range(2, 10):
            ax.axvline(n*period, alpha=0.4, lw=1, linestyle="dashed")
            ax.axvline(period / n, alpha=0.4, lw=1, linestyle="dashed")
        plt.ylabel(r'SDE BLS')
        plt.xlabel('Period (days)')
        plt.plot(results_bls.period, results_bls.power, color='black', lw=0.5)
        plt.xlim(0, max(results_bls.period))

    def plot_transit(self, results_bls):
        index = np.argmax(results_bls.power)
        period = results_bls.period[index]
        t0 = results_bls.transit_time[index]
        duration = results_bls.duration[index]

        plt.figure()
        ax = plt.gca()
        x = (self.t - t0 + 0.5*period) % period - 0.5*period
        m = np.abs(x) < 0.5
        plt.scatter(
            x[m],
            self.y_filt[m],
            color='blue',
            s=10,
            alpha=0.5,
            zorder=2)
        x = np.linspace(-0.13, 0.13, 1000)
        model_bls = BoxLeastSquares(self.t, self.y_filt)
        f = model_bls.model(x + t0, period, duration, t0)
        ax.plot(x, f, color='red')
        #ax.set_xlim(-0.17, 0.17)
        #plt.ylim(0.9985, 1.00025)
        ax.set_xlabel("Time from mid-transit (days)")
        ax.set_ylabel("Flux")



class GaiaDataFetcher:
    def __init__(self, tic_id):
        self.tic_id = tic_id

    def fetch_gaia_data(self):
        catalog_data = Catalogs.query_criteria(catalog='Tic', ID=self.tic_id)
        ra = catalog_data['ra'][0]
        dec = catalog_data['dec'][0]

        gaia_result = Catalogs.query_region(coordinates=f'{ra} {dec}', radius=0.001, catalog='Gaia')
        source_id = gaia_result['source_id'][0]

        query = f"SELECT radial_velocity, radial_velocity_error, ref_epoch FROM gaiadr3.gaia_source WHERE source_id = {source_id}"
        result = Gaia.launch_job(query).get_results()

        velocidad_radial = result['radial_velocity']
        error_velocidad_radial = result['radial_velocity_error']
        time_vr = result['ref_epoch']
        dispersion_velocidad_radial = np.std(velocidad_radial)

        return velocidad_radial, error_velocidad_radial



class TLS_model:
    def __init__(self, url, tic_id):
        self.url = url
        self.tic_id = tic_id

    def process_data(self):
        hdu = fits.open(self.url)
        time = hdu[1].data['TIME']
        flux = hdu[1].data['PDCSAP_FLUX']
        time, flux = cleaned_array(time, flux)  
        flux /= np.median(flux)
        print('Flux median:', np.median(flux))

        TIC_ID = hdu[0].header['TICID']
        ab, _, _, _, _, _, _ = catalog_info(TIC_ID=self.tic_id)
        #ab, mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(TIC_ID=self.tic_id)

        print('Searching with limb-darkening estimates using quadratic LD (a,b)=', ab)

        model = transitleastsquares(time, flux)
        results = model.power(u=ab)

        self.plot_periodogram(results)

        return time, flux, results

    def plot_periodogram(self, results):
        plt.figure()
        ax = plt.gca()
        ax.axvline(results.period, alpha=0.4, lw=3)

        plt.xlim(np.min(results.periods), np.max(results.periods))
        for n in range(2, 10):
            ax.axvline(n*results.period, alpha=0.4, lw=1, linestyle="dashed")
            ax.axvline(results.period / n, alpha=0.4, lw=1, linestyle="dashed")
        plt.ylabel(r'SDE')
        plt.xlabel('Period (days)')
        plt.plot(results.periods, results.power, color='black', lw=0.5)
        plt.xlim(0, max(results.periods))
        plt.show()
        
    def plot_phase(self, results):
        plt.figure()
        plt.plot(results.model_folded_phase, results.model_folded_model, color='red')
        plt.scatter(results.folded_phase, results.folded_y, color='blue', s=10, alpha=0.5, zorder=2)
        #plt.xlim(0.48, 0.52)
        plt.xlabel('Phase')
        plt.ylabel('Relative flux')
        plt.show()
