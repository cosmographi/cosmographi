from typing import Optional

from caskade import Param, forward, active_cache
import jax
import jax.numpy as jnp

from .base import MagSystem
from ..throughput.base import Throughput
from ..utils import flux


class MagAB(MagSystem):
    def __init__(self, throughput: Optional[Throughput] = None, name=None):
        super().__init__(name=name)
        self.throughput = throughput

    @active_cache
    @forward
    def _get_flux_refs(self):
        bands = jnp.arange(len(self.throughput.bands))
        # AB magnitude system reference flux in erg/s/cm^2/Hz
        f_nu = 3631 * 1e-23  # Convert Jy to erg/s/cm^2/Hz
        nu = flux.nu(self.throughput.w)
        return jax.vmap(flux.f_lambda_band)(
            self.throughput.w,
            flux.f_l(nu, f_nu),
            jax.vmap(self.throughput.T)(self.throughput.w, bands),
        )

    @forward
    def reference_flux(self, band: int):
        """
        Return the normalization flux for the AB magnitude system and given
        throughput. This is computed assuming a flat spectrum in f_nu
        (erg/s/cm^2/Hz) of 3631 Jy, which is the standard reference flux for AB
        magnitudes.

        Parameters
        ----------
        band : int
            Index of the filter band to use for the flux normalization calculation.

        Returns
        -------
        flux_ref : jnp.ndarray
            Reference flux for the filter in the AB magnitude system in (electrons/s/cm^2).
        """
        flux_refs = self._get_flux_refs()
        return flux_refs[band]


class MagZP(MagSystem):
    def __init__(self, zp=None, gain=None, name=None):
        super().__init__(name=name)
        self.zp = Param("zp", zp, shape=(None,), description="Magnitude zero point")
        self.gain = Param(
            "gain",
            gain,
            shape=(None,),
            description="Gain conversion factor between fluxes from 10^(-0.4(mag - zp)) to electrons/s/cm^2",
            units="electrons/ADU",
        )

    @forward
    def reference_flux(self, band: int, zp):
        """
        Return the reference flux for the zero point magnitude system and given
        throughput. This is computed using the zero point magnitude (zp) and gain.

        Parameters
        ----------
        band : int
            Index of the filter band to use for the flux normalization calculation.

        Returns
        -------
        flux_ref : jnp.ndarray
            Reference flux for each filter in the zero point magnitude system in (electrons/s/cm^2).
        """
        return 10 ** (0.4 * zp[band])

    @forward
    def __call__(self, band, fluxes: jnp.ndarray, zp):
        """
        Convert fluxes to magnitudes using the typical magnitude formula:

        m = -2.5 * log10(flux / flux_ref)

        where flux_ref is calculated using the zero point magnitude (zp) and gain.

        Parameters
        ----------
        band : int
            Index of the filter band to use for the magnitude conversion.
        fluxes : jnp.ndarray
            Fluxes in the magnitude system (flux / flux_ref) for each filter.
        zp : jnp.ndarray
            Zero point magnitudes for each filter.

        Returns
        -------
        mags : jnp.ndarray
            Magnitudes corresponding to the input fluxes.
        """
        # Calculate the normalization flux using the zero point magnitude and gain
        return -2.5 * jnp.log10(fluxes) + zp[band]
