import jax.numpy as jnp
from .base import Instrument
from ..throughput import Throughput
from ..throughput.base import Throughput_wAtmos
from ..magsystem import MagSystem, MagAB


class RubinObservatory(Instrument):
    def __init__(
        self,
        throughput: Throughput = Throughput_wAtmos.load("rubin_throughput"),
        mag_system: MagSystem = MagAB(),
        name=None,
        **kwargs,
    ):

        Aeff = jnp.pi * (649 / 2) ** 2  # Effective area of the Rubin Observatory in cm^2
        super().__init__(throughput, Aeff=Aeff, mag_system=mag_system, name=name, **kwargs)
