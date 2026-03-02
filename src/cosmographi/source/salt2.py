import os

import jax.numpy as jnp
import jax
from caskade import Param, forward
import numpy as np

from .base import TransientSource
from ..utils import load_salt2_surface, load_salt2_colour_law, flux
from ..utils.constants import Mpc_to_cm


class SALT2_2021(TransientSource):
    SALT2CL_B = 430.257  # B-band-ish wavelength (nm, rest frame)
    SALT2CL_V = 542.855  # V-band-ish wavelength (nm, rest frame)

    # fixme figure out what to do about alpha and beta
    def __init__(
        self,
        cosmology=None,
        x1=None,
        c=None,
        M=None,
        CL=None,
        phase_nodes=None,
        wavelength_nodes=None,
        name=None,
        **kwargs,
    ):
        super().__init__(cosmology=cosmology, name=name, **kwargs)
        self.x1 = Param("x1", x1, shape=(), description="Light curve stretch")
        self.c = Param("c", c, shape=(), description="colour")
        self.M = Param(
            "M",
            M,
            shape=(2, None, None),
            dynamic=False,
            description="Phase and wavelength dependent SALT2 model surface M0 and M1 components (luminosity density)",
        )
        self.CL = Param(
            "CL", CL, shape=(None,), dynamic=False, description="Phase independent SALT2 colour law"
        )

        self.phase_nodes = phase_nodes  # Phase nodes of M
        self.wavelength_nodes = wavelength_nodes  # wavelength nodes of M and CL
        self.phase_sampler = jax.vmap(jnp.interp, in_axes=(None, None, 1), out_axes=-1)

    def min_phase(self):
        return self.phase_nodes[0]

    def max_phase(self):
        return self.phase_nodes[-1]

    @forward
    def get_model_basis(self, p, M):
        return tuple(self.phase_sampler(p, self.phase_nodes, M[i]) for i in range(M.shape[0]))

    @forward
    def colour_law(self, w, CL):
        w_mod = (w - self.SALT2CL_B) / (self.SALT2CL_V - self.SALT2CL_B)
        CL = jnp.concatenate(
            (
                jnp.zeros(1),
                1 - jnp.sum(CL)[None],
                CL,
            )
        )  # alpha is 1 - sum of the other coefficients
        CL_edge_vals = jnp.polyval(CL[::-1], self.CL_wrange)
        dCL_edge_vals = jnp.polyval((CL * jnp.arange(len(CL)))[1:][::-1], self.CL_wrange)
        corr = jnp.where(
            w_mod < self.CL_wrange[0],
            CL_edge_vals[0]
            + dCL_edge_vals[0] * (w_mod - self.CL_wrange[0]),  # Low edge, linear extrapolation
            jnp.where(
                w_mod > self.CL_wrange[1],
                CL_edge_vals[1]
                + dCL_edge_vals[1] * (w_mod - self.CL_wrange[1]),  # High edge, linear extrapolation
                jnp.polyval(CL[::-1], w_mod),  # Within edges, polynomial evaluation
            ),
        )
        return corr

    @forward
    def luminosity_density(self, w, p, t0, x1, c, z):
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
        p0 = flux.observer_to_rest_time(t0, z)
        p = p - p0
        M0, M1 = self.get_model_basis(p)
        f_l = M0 + x1 * M1
        F = jnp.interp(w, self.wavelength_nodes, f_l) * jnp.exp(c * self.colour_law(w))
        return jnp.where(p < self.min_phase(), 0.0, jnp.where(p > self.max_phase(), 0.0, F))

    def load_salt2_model(self, directory=None):
        if directory is None:
            directory = os.path.join(os.path.dirname(__file__), "data/SALT2_2021/")
        phase0, wavelength0, M0 = load_salt2_surface(
            os.path.join(directory, "salt2_template_0.dat")
        )

        phase1, wavelength1, M1 = load_salt2_surface(
            os.path.join(directory, "salt2_template_1.dat")
        )

        assert np.all(phase0 == phase1), "Phase gridding does not match for M0 and M1!"
        assert np.all(wavelength0 == wavelength1), (
            "Wavelength gridding does not match for M0 and M1!"
        )
        # convert from spectral flux density to luminosity density (should just be 4 * pi * 10pc^2)
        self.M = (
            np.stack((M0, M1)) * (4 * np.pi * (10 * 1e-6 * Mpc_to_cm) ** 2) * 10
        )  # x10 to convert from /Angstrom to /nm
        self.phase_nodes = jnp.array(phase0)
        # Convert from angstroms to nm
        self.wavelength_nodes = jnp.array(wavelength0) / 10

        CL_coefs, lmin, lmax = load_salt2_colour_law(
            os.path.join(directory, "salt2_color_correction.dat")
        )
        self.CL_wrange = jnp.array(
            (
                (lmin / 10 - self.SALT2CL_B) / (self.SALT2CL_V - self.SALT2CL_B),
                (lmax / 10 - self.SALT2CL_B) / (self.SALT2CL_V - self.SALT2CL_B),
            )
        )
        self.CL = jnp.array(CL_coefs)
