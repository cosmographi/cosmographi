from typing import Type
from .base import Instrument
from .effects.base import InstrumentEffect


def instrument_factory(
    base_instrument: Type[Instrument], *effects: tuple[Type[InstrumentEffect]]
) -> Type[Instrument]:
    """
    Creates a new class that inherits from the effects (mixins) and the base
    instrument class.

    Organize the effects as though the photons are moving left to right, from
    the "base_instrument" to the observer on the right.
    """
    assert issubclass(base_instrument, Instrument), "base_instrument should be an Instrument model"
    assert all(issubclass(eff, InstrumentEffect) for eff in effects), (
        "Effects should be InstrumentEffects"
    )

    # Name the class dynamically for better debugging
    name = f"{base_instrument.__name__}"
    for effect in effects:
        name += f"_{effect.__name__}"

    new_instrument = type(name, (*reversed(effects), base_instrument), {})

    return new_instrument
