from typing import Optional
from abc import abstractmethod
from caskade import Module, forward, Param
import jax.numpy as jnp

from ..cosmology import Cosmology
from ..cosmology.func import mu_to_luminosity_distance
from ..utils import flux
from ..utils.constants import c_nm


class Source(Module):
    """
    Base class for representing astronomical sources.

    This class serves as a foundation for more specific source types, such as stars,
    galaxies, or supernovae. It can be extended to include common properties and methods
    relevant to all source types.
    """

    def __init__(
        self,
        cosmology: Optional[Cosmology] = None,
        z: Optional[float] = None,
        mu: Optional[float] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.z = Param("z", z, shape=(), description="Redshift", units="dimensionless")
        self.mu = Param("mu", mu, shape=(), description="Distance modulus", units="magnitudes")
        if cosmology is not None and mu is None:
            self.link_cosmology(cosmology)

    def link_cosmology(self, cosmology: Cosmology):
        self.mu = lambda p: p.cosmology.distance_modulus(p.z.value)
        self.mu.link(["cosmology", "z"], [cosmology, self.z])

    @abstractmethod
    def luminosity_density(self, w: jnp.ndarray, *args, **kwargs):
        raise NotImplementedError("Please use a subclass of Source")

    @abstractmethod
    def spectral_flux_density(self, w: jnp.ndarray, *args, **kwargs):
        raise NotImplementedError("Please use a subclass of Source")


class StaticSource(Source):
    """
    Class for representing static astronomical sources.

    This class is intended for sources that do not vary over time, such as stars or galaxies.
    It can be extended to include properties and methods specific to static sources.
    """

    @abstractmethod
    def luminosity_density(self, w: jnp.ndarray):
        """
        Calculate the luminosity density at a given wavelength in units of erg/s/nm.
        """
        raise NotImplementedError("Subclasses must implement the luminosity_density method.")

    @forward
    def spectral_flux_density(self, w: jnp.ndarray, z: jnp.ndarray, mu: jnp.ndarray):
        ld = self.luminosity_density(w)
        DL = mu_to_luminosity_distance(mu) / 1e6  # Convert pc to Mpc
        return flux.f_lambda(z, DL, ld)

    @forward
    def spectral_flux_density_frequency(self, nu: jnp.ndarray):
        w = c_nm / nu
        f_l = self.spectral_flux_density(w)
        return flux.f_nu(w, f_l)


class TransientSource(Source):
    """
    Class for representing transient astronomical sources.

    This class is intended for sources that vary over time, such as supernovae or variable stars.
    It can be extended to include properties and methods specific to transient sources.
    """

    def __init__(
        self,
        cosmology: Optional[Cosmology] = None,
        z: Optional[float] = None,
        t0: Optional[float] = None,
        name: Optional[str] = None,
    ):
        super().__init__(cosmology, z=z, name=name)

        self.t0 = Param(
            "t0",
            t0,
            shape=(),
            description="Light curve reference time (observer frame)",
            units="days",
        )

    @abstractmethod
    def min_phase(self):
        raise NotImplementedError("Subclasses must implement the min_phase method.")

    @abstractmethod
    def max_phase(self):
        raise NotImplementedError("Subclasses must implement the max_phase method.")

    @forward
    def min_time(self, z, t0):
        return t0 + flux.rest_to_observer_time(self.min_phase(), z)

    @forward
    def max_time(self, z, t0):
        return t0 + flux.rest_to_observer_time(self.max_phase(), z)

    @forward
    def visible(self, t, t0, z):
        t_range = (
            flux.rest_to_observer_time(self.min_phase(), z),
            flux.rest_to_observer_time(self.max_phase(), z),
        )
        return (t >= t0 + t_range[0]) & (t <= t0 + t_range[1])

    @abstractmethod
    def luminosity_density(self, w: jnp.ndarray, p: jnp.ndarray):
        """
        Calculate the luminosity density at a given wavelength in units of
        erg/s/nm and time in units of seconds.

        Parameters
        ----------
        w : jnp.ndarray
            Wavelength array (nm) rest frame
        p : jnp.ndarray
            Time (phase) of observation (days) rest frame
        """
        raise NotImplementedError("Subclasses must implement the luminosity_density method.")

    @forward
    def spectral_flux_density(
        self, w: jnp.ndarray, t: jnp.ndarray, z: jnp.ndarray, mu: jnp.ndarray
    ):
        """
        Calculate the observed spectral flux density at a given redshift z.
        This method should account for z effects on the spectral flux density.

        Here we use:

        $$f_{\\lambda}^{obs}(\\lambda_{obs}) = \\frac{1}{(1 + z) D_L^2}L_{\\lambda}^{rest}(\\lambda_{obs} / (1 + z))$$

        where $D_L$ is the luminosity distance in cm. $\\lambda_{obs}$ is the
        observed wavelength. $f_{\\lambda}^{obs}$ is the observed spectral flux density (erg/s/cm^2/nm), and
        $L_{\\lambda}^{rest}$ is the rest-frame luminosity density (erg/s/nm).

        Parameters
        ----------
        w : jnp.ndarray
            Wavelength array (nm) observer frame.
        t : jnp.ndarray or float
            Time of observation (days) observer frame

        Returns
        -------
        spectral_flux_density : jnp.ndarray
            Spectral flux density in (erg/s/cm^2/nm)
        """
        w = flux.observer_to_rest_wavelength(w, z)
        p = flux.observer_to_rest_time(t, z)
        ld = self.luminosity_density(w, p)
        DL = mu_to_luminosity_distance(mu) / 1e6  # Convert pc to Mpc
        return flux.f_lambda(z, DL, ld)

    @forward
    def spectral_flux_density_frequency(self, nu: jnp.ndarray, t: jnp.ndarray):
        w = c_nm / nu
        f_l = self.spectral_flux_density(w, t)
        return flux.f_nu(w, f_l)
