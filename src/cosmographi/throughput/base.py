import jax.numpy as jnp
from caskade import Module, Param, forward

from ..utils.helpers import trim_and_pad_batch
from ..utils import flux
from ..utils.registry import load_data


class Throughput(Module):
    """Stores throughput curves

    These combine bandpass filters, atmosphere throughput, mirror/lens and other
    hardware throughput.

    This object holds the wavelength and transmission curves for batched filters
    including all effects between the atmosphere and the CCD. The primary
    attributes are w and T. w is the wavelength (nm) array corresponding to the
    wavelengths at which the transmissions are evaluated. T is the transmission
    array, the first dimension indexes the filters, the second gives the
    transmission corresponding to the wavelength array.

    Parameters
    ----------
    names : list of str
        Names of the filters in the throughput.
    w : array-like
        Wavelength array (nm) for the throughput curves. Can be 1D or 2D. If 1D, it is assumed to be the same for all filters.
    T : array-like
        Transmission array for the throughput curves. Should be 2D with shape (n_filters, n_wavelengths).
    """

    def __init__(
        self,
        bands: list[str],
        w: jnp.ndarray,
        T: jnp.ndarray,
        name=None,
    ):
        super().__init__(name=name)
        self.bands = bands
        if T.ndim == 1:
            T = T[None]
        if w.ndim == 1:
            w = w.reshape(1, -1).repeat(T.shape[0], axis=0)
        self._w = w
        self._T = T

    def T(self, w, b):
        """Total transmission including hardware and atmosphere."""
        return jnp.interp(w, self._w[b], self._T[b])

    def T_nu(self, nu, b):
        """Total transmission in frequency space."""
        return self.T(flux.w(nu), b)

    @property
    def w(self):
        return self._w

    @property
    def nu(self):
        """Frequency array corresponding to the wavelength array."""
        return flux.nu(self.w[:, ::-1])

    def trim(self, threshold: float = 1e-4):
        """Trim the wavelength range for each filter to where the transmission is above a threshold."""

        self._w, self._T = trim_and_pad_batch(self._w, self._T, threshold)


class Throughput_wAtmos(Throughput):
    def __init__(
        self, bands, w_hardware, T_hardware, w_atmosphere, T_atmosphere, air_mass=1.0, name=None
    ):
        super().__init__(bands=bands, w=w_hardware, T=T_hardware, name=name)
        self.w_atmosphere = w_atmosphere
        self.T_atmosphere = T_atmosphere
        self.air_mass = Param(
            "air_mass",
            air_mass,
            shape=(),
            description="Scaling for strength of atmosphere transmission (T ^ air_mass)",
            units="unitless",
        )

    @classmethod
    def load(cls, key: str, *args, **kwargs):
        data = load_data(key)
        _kwargs = data | kwargs
        return cls(*args, **_kwargs)

    @forward
    def T(self, w, b, air_mass):
        return super().T(w, b) * jnp.interp(w, self.w_atmosphere, self.T_atmosphere**air_mass)
