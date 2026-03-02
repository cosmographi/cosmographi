import jax
import jax.numpy as jnp
from caskade import Module, forward

from ..throughput import Throughput
from ..magsystem import MagSystem
from ..source import Source
from ..utils import flux


class Instrument(Module):
    """
    Base Instrument object that combines a throughput and magnitude system to
    simulate observations of sources.

    This class handles essentially everything that happens to the source
    spectrum from the top of the atmosphere to the CCD (camera). It takes in a
    source spectrum and produces an observed flux and magnitude for a given
    filter, including noise. The main components of the instrument are the
    throughput (which includes the effects of the atmosphere, telescope optics,
    and filters) and the magnitude system (which defines how fluxes are
    normalized and converted to magnitudes).
    """

    def __init__(
        self,
        throughput: Throughput,
        mag_system: MagSystem,
        dark_current=0.0,
        read_noise=0.0,
        effective_aperture=None,
        pixelscale=None,
        fov=None,
        name=None,
    ):
        super().__init__(name=name)
        self.throughput = throughput
        self.mag_system = mag_system
        self.dark_current = dark_current  # Rate of electrons that accumulate in a pixel while dark, in electrons/s/pixel
        self.read_noise = read_noise  # Standard deviation of electron counts per pixel due to readout noise, in RMS electrons/pixel
        self.effective_aperture = effective_aperture  # Effective telescope aperture size in cm^2
        self.pixelscale = pixelscale  # Pixel scale in arcsec/pixel
        self.fov = fov

    def _observe_spectrum(self, band, source: Source, *args, **kwargs):
        return source.spectral_flux_density(self.throughput.w[band], *args, **kwargs)

    @forward
    def flux(self, band, source: Source, *args, **kwargs):
        """
        Return the flux of a source observed through the instrument's
        throughput. The result is provided in electrons/s/cm^2 rather than being
        normalized by a magnitude system.

        Parameters
        ----------
        band : int
            Index of the filter band to use for the flux calculation.
        source : BaseSource
            The source for which to calculate the flux.
        *args, **kwargs
            Additional arguments to pass to the source's spectral_flux_density
            method. Note that the wavelength argument

        Returns
        -------
        flux : float
            The observed flux of the source through the specified filter in
            electrons/s/cm^2, given the observing conditions.
        """
        f = self._observe_spectrum(band, source, *args, **kwargs)
        w = self.throughput.w[band]
        F = flux.f_lambda_band(w, f, self.throughput.T(band, w))
        return F

    @forward
    def observation_var(self, band, exp_time, sky_brightness, PSF_Aeff) -> float:
        """
        Return the variance on an observation of a source. The variance is
        computed as the expected number of electrons from the noise source.

        Returns
        -------
        var : float
            The variance on an observation of a point source in electron counts.
        """
        # Sky brightness noise in electrons
        sky_flux = self.mag_system.mag_to_electron_flux(band, sky_brightness * PSF_Aeff)
        sky_Ne = sky_flux * exp_time

        # Dark current noise in electrons
        PSF_pixel_Aeff = PSF_Aeff / self.pixelscale**2  # Effective area of the PSF in pixels
        dark_Ne = self.dark_current * exp_time * PSF_pixel_Aeff

        # Read noise in electrons
        read_noise = self.read_noise**2 * PSF_pixel_Aeff

        return sky_Ne + dark_Ne, read_noise

    @forward
    def observe(
        self, key, band, exp_time, sky_brightness, PSF_Aeff, source: Source, *args, **kwargs
    ):
        """
        Create a mock observation of a source through the instrument's
        throughput, including noise.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key for generating noise in the observation.
        band : int
            Index of the filter band to use for the observation.
        exp_time : float
            Image exposure time in seconds.
        sky_brightness : float
            Sky brightness in mag/arcsec^2.
        PSF_Aeff : float
            Effective area of the PSF in arcsec^2.
        source : BaseSource
            The source to observe.
        *args, **kwargs
            Additional arguments to pass to the source's spectral_flux_density
            method. Note that the wavelength argument (w) is provided by the
            Throughput object and should not be passed in by the user.

        Returns
        -------
        flux_obs : float
            The observed flux of the source through the specified filter,
            normalized by the magnitude system's reference flux, including
            noise.
        flux_err_obs : float
            The observed uncertainty on the flux of the source through the
            specified filter, normalized by the magnitude system's reference
            flux, including noise.
        flux_true : float
            The true flux of the source through the specified filter, normalized
            by the magnitude system's reference flux, without noise.
        flux_err_true : float
            The true uncertainty on the flux of the source through the specified
            filter, normalized by the magnitude system's reference flux, without
            noise.

        Note
        ----
        The noise reported by this function is the **measured** noise, meaning
        the observed flux is converted to a number of electrons and the square
        root of the observed number of electrons is used as the measured noise.
        The actual noise generating process comes from the square root on the
        true number of expected electrons. If this is not clear, see the code
        with comments that explain the exact process.
        """
        # True flux in electrons/s/cm^2
        flux = self.flux(band, source, *args, **kwargs)
        # Scale factor between flux and number of electrons
        norm = exp_time * self.effective_aperture

        N = flux * norm  # Expected number of electrons
        Ve = jnp.abs(N)  # Flux error in electrons
        # Non-source contributions to noise variance in electrons
        Vbkg, Vread = self.observation_var(band, exp_time, sky_brightness, PSF_Aeff)

        # Observed number of electrons with noise
        noise = jax.random.normal(key, shape=flux.shape)
        Nobs = N + noise * jnp.sqrt(Ve + Vbkg + Vread)

        flux_obs = Nobs / norm  # Convert back to flux units
        # Measured flux uncertainty, set floor at Vread since typical variance plane
        # values are: Nelectrons + Vread, plus further calibration not modelled here
        flux_err_obs = jnp.sqrt(jnp.clip(Nobs + Vbkg + Vread, min=Vread)) / norm
        return flux_obs, flux_err_obs, flux, jnp.sqrt(Ve + Vbkg + Vread) / norm
