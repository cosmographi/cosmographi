import cosmographi as cg
import jax.numpy as jnp
import jax
from astropy.cosmology import wCDM

import pytest


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "Omega_m, Omega_k, H0, w0",
    [
        (0.2, 0.0, 70, -1),  # Flat LCDM
        (0.3, 0.1, 65, -1.1),  # Open LCDM
        (0.4, -0.1, 75, -1),  # Closed LCDM
        (0.5, 0.0, 71, -0.8),  # Flat wCDM
    ],
)
def test_distances(Omega_m, Omega_k, H0, w0):
    ap_cosmo = wCDM(H0=H0, Om0=Omega_m, Ode0=1 - Omega_m - Omega_k, w0=w0)
    cg_cosmo = cg.Cosmology(H0=H0, Omega_m=Omega_m, Omega_k=Omega_k, Omega_r=0.0, w0=w0, wa=0.0)

    z = jnp.linspace(0, 10, 1000)

    assert jnp.allclose(
        jax.vmap(cg_cosmo.comoving_distance)(z), ap_cosmo.comoving_distance(z), rtol=1e-9, atol=0
    )

    assert jnp.allclose(
        jax.vmap(cg_cosmo.luminosity_distance)(z),
        ap_cosmo.luminosity_distance(z),
        rtol=1e-9,
        atol=0,
    )

    assert jnp.allclose(
        jax.vmap(cg_cosmo.angular_diameter_distance)(z),
        ap_cosmo.angular_diameter_distance(z),
        rtol=1e-9,
        atol=0,
    )

    z = jnp.linspace(0, 10, 10)
    V_cg = jax.vmap(cg_cosmo.differential_comoving_volume)(z)
    V_ap = ap_cosmo.differential_comoving_volume(z)
    assert jnp.allclose(
        V_cg,
        V_ap,
        rtol=1e-8,
        atol=0,
    )
