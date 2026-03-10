import jax.numpy as jnp
from .base import Instrument
from ..throughput import Throughput
from ..throughput.base import Throughput_wAtmos
from ..magsystem import MagSystem, MagAB


class RubinObservatory(Instrument):
    def __init__(
        self,
        throughput: Throughput | None = None,
        mag_system: MagSystem | None = None,
        name=None,
        **kwargs,
    ):
        if throughput is None:
            throughput = Throughput_wAtmos.load("rubin_throughput")
        if mag_system is None:
            mag_system = MagAB(
                throughput=throughput,
                Ze=[
                    26.52,
                    28.51,
                    28.36,
                    28.17,
                    27.78,
                    26.82,
                ],  # Fixme, these should be calculated without atmosphere, see: https://smtn-002.lsst.io/
            )
        effective_aperture = jnp.pi * (649 / 2) ** 2  # Effective area in cm^2
        super().__init__(
            throughput,
            effective_aperture=effective_aperture,
            mag_system=mag_system,
            name=name,
            read_noise=8.8,  # electrons/pixel
            dark_current=0.2,  # electrons/s/pixel
            pixelscale=0.2,  # arcsec/pixel
            **kwargs,
        )
