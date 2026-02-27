from abc import abstractmethod

import jax.numpy as jnp
from caskade import Module, Param, forward


class MagSystem(Module):
    def __init__(self, Ze=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.Ze = Param(
            "Ze",
            Ze,
            shape=(None,),
            description="Electron zero point, conversion factor to take magnitudes into fluxes of electrons/s. Sometimes called instrumental zeropoint or instrumental magnitude.",
            units="emags",
        )

    @abstractmethod
    def reference_flux(self, band):
        """
        Return the normalization flux for the magnitude system in a given band.
        This is used to convert fluxes to magnitudes using:

        m = -2.5 * log10(flux / flux_ref)

        Subclasses must implement this method to provide the appropriate
        reference flux for their magnitude system. The reference flux is
        typically defined based on the specific filters and calibration used in
        the magnitude system, and it serves as the baseline for converting
        observed fluxes into magnitudes.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def flux_to_mag(self, band, flux: jnp.ndarray):
        """
        Convert fluxes to magnitudes using the typical magnitude formula:

        m = -2.5 * log10(flux / flux_ref)

        where flux_ref is calculated using the reference_flux method.

         Subclasses can override this method if they have a different formula for
        converting fluxes to magnitudes, but the default implementation assumes
        the standard logarithmic relationship used in most magnitude systems.
        The reference_flux parameter allows for flexibility in how the input fluxes
        are normalized before conversion to magnitudes. If reference_flux is not
        provided, it is assumed that the input fluxes are already normalized
        appropriately for the magnitude system being used.

         Parameters
        ----------
        band : int
            Index of the filter band to use for the magnitude conversion.
        flux : jnp.ndarray
            Fluxes in the magnitude system for each filter.

        Returns
        -------
        mags : jnp.ndarray
            Magnitudes corresponding to the input fluxes.
        """
        flux_ref = self.reference_flux(band)
        return -2.5 * jnp.log10(flux / flux_ref)

    def mag_to_flux(self, band, mags: jnp.ndarray):
        """
        Convert magnitudes to fluxes using the reference flux for the magnitude system.

        Parameters
        ----------
        band : int
            Index of the filter band to use for the flux conversion.
        mags : jnp.ndarray
            Magnitudes in the magnitude system for each filter.

        Returns
        -------
        fluxes : jnp.ndarray
            Fluxes corresponding to the input magnitudes in electrons/s/cm^2.
        """
        flux_ref = self.reference_flux(band)
        return flux_ref * 10 ** (-0.4 * mags)

    @forward
    def mag_to_electron_flux(self, band, mag: jnp.ndarray, Ze):
        """
        Convert magnitudes to fluxes in electrons/s using the electron zeropoint
        for the magnitude system.

        Parameters
        ----------
        band : int
            Index of the filter band to use for the flux conversion.
        mag : jnp.ndarray
            Magnitudes in the magnitude system for each filter.

        Returns
        -------
        fluxes : jnp.ndarray
            Fluxes corresponding to the input magnitudes in electrons/s.
        """
        return 10 ** (-0.4 * (mag - Ze[band]))

    def err(self, flux: jnp.ndarray, flux_err: jnp.ndarray):
        """
        Convert flux errors to magnitude errors using error propagation for the typical magnitude formula.

        Parameters
        ----------
        flux : jnp.ndarray
            Fluxes in the magnitude system for each filter.
        flux_err : jnp.ndarray
            Errors on the fluxes in the magnitude system for each filter.

        Returns
        -------
        mag_err : jnp.ndarray
            Magnitude errors corresponding to the input flux errors.
        """
        # Error propagation for m = -2.5 * log10(flux)
        return 2.5 * (flux_err / flux) / jnp.log(10)
