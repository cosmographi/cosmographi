import cosmographi as cp
import jax.numpy as jnp
import pytest


@pytest.mark.parametrize("air_mass", [0.9, 1.0, 1.2])
def test_MagAB(air_mass, benchmark_jax):
    T = cp.RubinThroughput(air_mass=air_mass)
    mag_system = cp.MagAB(throughput=T, Ze=jnp.ones(len(T.bands)) * 25)
    C = cp.Cosmology()
    BB = cp.StaticBlackbody(cosmology=C, z=1.0, T=5000, R=6e10, N=1e9)
    mag_system.reference_flux(0, T)
    f = BB.spectral_flux_density(T.w[0])
    F = cp.utils.flux.f_lambda_band(T.w[0], f, T.T(0, T.w[0]))
    mag = benchmark_jax(mag_system.flux_to_mag, 0, F)
    assert jnp.isclose(mag, 29.80607931, atol=1e-2, rtol=0), (
        "AB magnitude of T=5000K blackbody should be 29.80607931."
    )
