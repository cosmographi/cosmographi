import jax.numpy as jnp
import pytest

from cosmographi import MagAB, Cosmology, StaticBlackbody
from cosmographi.throughput.base import Throughput_wAtmos
from cosmographi.utils.flux import f_lambda_band


@pytest.mark.parametrize("air_mass", [0.9, 1.0, 1.2])
def test_MagAB(air_mass, benchmark_jax):
    T = Throughput_wAtmos.load("rubin_throughput", air_mass=air_mass)
    mag_system = MagAB(throughput=T, Ze=jnp.ones(len(T.bands)) * 25)
    C = Cosmology()
    BB = StaticBlackbody(cosmology=C, z=1.0, T=5000, R=6e10, N=1e9)
    mag_system.reference_flux(0, T)
    f = BB.spectral_flux_density(T.w[0])
    F = f_lambda_band(T.w[0], f, T.T(0, T.w[0]))
    mag = benchmark_jax(mag_system.flux_to_mag, 0, F)
    assert jnp.isclose(mag, 29.80607931, atol=1e-2, rtol=0), (
        "AB magnitude of T=5000K blackbody should be 29.80607931."
    )
