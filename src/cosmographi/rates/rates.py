from caskade import Param, forward
import jax.numpy as jnp

from .base import Rate


class RateConst(Rate):
    """
    Constant event rate module.
    """

    def __init__(self, cosmology, r, **kwargs):
        super().__init__(cosmology, **kwargs)
        self.r = Param(
            "r",
            r,
            description="event rate per unit volume",
            units="1/yr/Mpc^3",
        )

    @forward
    def rate_density(self, z, r=None):
        """
        Calculate the event rate at redshift z.
        """
        return r


class RateInterp(Rate):
    """
    1D linear interpolating event rate module.
    """

    def __init__(self, cosmology, rate_z_nodes, r, **kwargs):
        super().__init__(cosmology, **kwargs)
        self.rate_z_nodes = jnp.array(rate_z_nodes)
        self.r = Param(
            "r",
            r,
            description="Event rate per unit volume at redshift z",
            units="1/yr/Mpc^3",
        )

    @forward
    def rate_density(self, z, r=None):
        """
        Calculate the event rate at redshift z.
        """
        # Placeholder for actual calculation
        return jnp.interp(z, self.rate_z_nodes, r)
