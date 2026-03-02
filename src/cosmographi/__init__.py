__version__ = "0.0.0"

import jax

jax.config.update("jax_enable_x64", True)
import caskade as ck

ck.backend.backend = "jax"


from . import (
    # sn,
    detect,
    survey,
    # likelihood,
    sims,
    utils,
)
from .rates import Rate, RateConst, RateInterp, CombinedRate
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
from .throughput import Throughput, RubinThroughput
from .magsystem import MagSystem, MagAB
from .cosmology import Cosmology

__all__ = (
    "sn",
    "detect",
    "rates",
    "Rate",
    "RateConst",
    "RateInterp",
    "CombinedRate",
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
    "RubinThroughput",
    "MagSystem",
    "MagAB",
    "Cosmology",
)
