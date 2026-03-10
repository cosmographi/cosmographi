__version__ = "0.0.0"
from . import (
    sn,
    detect,
    rates,
    survey,
    likelihood,
    sims,
    utils,
)
from .source import (
    Source,
    TransientSource,
    StaticSource,
    effects,
    SALT2_2021,
    source_factory,
    StaticBlackbody,
    TransientBlackbody,
)
from .instrument import Instrument, RubinObservatory
from .throughput import Throughput
from .magsystem import MagSystem, MagAB
from .cosmology import Cosmology

import jax

jax.config.update("jax_enable_x64", True)
import caskade as ck

ck.backend.backend = "jax"

__all__ = (
    "sn",
    "detect",
    "rates",
    "Source",
    "TransientSource",
    "StaticSource",
    "effects",
    "SALT2_2021",
    "source_factory",
    "StaticBlackbody",
    "TransientBlackbody",
    "survey",
    "likelihood",
    "sims",
    "utils",
    "Instrument",
    "RubinObservatory",
    "Throughput",
    "MagSystem",
    "MagAB",
    "Cosmology",
)
