import jax.numpy as jnp
from .base import Instrument
from ..throughput import Throughput, RubinThroughput
from ..magsystem import MagSystem, MagAB


class RubinObservatory(Instrument):
    def __init__(
        self,
        throughput: Throughput = RubinThroughput(),
        mag_system: MagSystem = MagAB(),
        name=None,
        **kwargs,
    ):
        effective_aperture = jnp.pi * (649 / 2) ** 2  # Effective area in cm^2
        super().__init__(
            throughput,
            effective_aperture=effective_aperture,
            mag_system=mag_system,
            name=name,
            read_noise=6.0,  # electrons/pixel
            dark_current=1.0 / 15,  # electrons/s/pixel
            pixelscale=0.2,  # arcsec/pixel
            **kwargs,
        )
