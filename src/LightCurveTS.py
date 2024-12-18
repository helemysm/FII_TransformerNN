import numpy as np
import lightkurve as lk
from periodicity_analysis import BLSP_model
import matplotlib.pyplot as plt

class LightCurveTS:
    def __init__(self, filename):
        self.filename = filename
        self._load_data()
        self._normalize_flux()
        self._calculate_period()
        self._print_results()

    def _load_data(self):
        self.tpf = lk.read(self.filename)
        self.lc = self.tpf.remove_nans().flatten()
        self.lc_time = self.lc.time.value
        self.lc_flux = self.lc.flux.value
        self.lc_flux_err = self.lc.flux_err.value

    def _normalize_flux(self):
        self.lc_flux = np.array(self.lc_flux) / np.median(np.array(self.lc_flux))

    def _calculate_period(self):
        bls_plotter = BLSP_model(self.lc_time, self.lc_flux)
        self.period, self.results_bls = bls_plotter.calculate_period()

        index = np.argmax(self.results_bls.power)
        self.period = self.results_bls.period[index]
        self.t0 = self.results_bls.transit_time[index]
        self.duration = self.results_bls.duration[index]
        self.depth = self.results_bls.depth[index]

    def _print_results(self):
        """Prints the calculated period, transit time, duration, and depth."""
        print("Period_ = ", self.period)
        print("bjd_ = ", self.t0)
        print("tdur_ = ", self.duration * 24)
        print("tdep_ = ", self.depth * 1000000)

    def get_transit_points(self, isplot=False, path=None):
        
        lc_timex = (self.lc_time - self.t0 + 0.5 * self.period) % self.period - 0.5 * self.period
        m = np.abs(lc_timex) < 0.5
        if isplot:
            self.plot_periodogram(lc_timex[m], self.lc_flux[m], self.period, path)
        
        return lc_timex[m], self.lc_flux[m], self.lc_flux_err[m], self.period, self.depth
    
    def plot_periodogram(self, lc_timex, lc_flux, period, path):
        
        plt.figure()
        ax = plt.gca()
        #lc_timex = (lc_time - t0 + 0.5*period) % period - 0.5*period
        #m = np.abs(lc_timex) < 0.5
        #phase_min = -0.25
        #phase_max = 0.25
        #m = (lc_timex >= phase_min) & (lc_timex <= phase_max)
        plt.scatter(
            lc_timex,
            lc_flux,
            color='blue',
            s=10,
            alpha=0.5,
            zorder=2)

        ax.set_xlabel("Time from mid-transit (days)")
        ax.set_ylabel("Flux")
        plt.savefig(path)

        