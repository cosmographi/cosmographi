"""
Cosmology utilities: Hubble parameter, comoving distance, and distance modulus, implemented in JAX.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import factorial
import cosmographi as cg

c_m = 299792458.0  # speed of light, m/s
c_km = c_m / 1000.0  # speed of light, km/s


c_m = 299792458  # speed of light in m/s
c_km = c_m / 1000  # speed of light in km/s


def H(
    z,
    H0=67.9,
    Omega_m=0.307,
    Omega_k=0,
    Omega_r=0,
    w0=-1.0,
    wa=0,
):
    """
    Calculate the Hubble parameter at redshift z. Units: km/s/Mpc.
    """
    Omega_l = 1 - Omega_m - Omega_k - Omega_r
    return (
        H0
        * (
            Omega_m * (1 + z) ** 3
            + Omega_k * (1 + z) ** 2
            + Omega_r * (1 + z) ** 4
            + Omega_l * (1 + z) ** (3 * (1 + w0 + wa * z / (1 + z)))
        )
        ** 0.5
    )


def comoving_distance(z, H0=67.9, Omega_m=0.307, Omega_k=0, Omega_r=0, w0=-1.0, wa=0):
    # The integrand must call the function H(z_prime)
    integrand = lambda zp: c_km / H(zp, H0, Omega_m, Omega_k, Omega_r, w0, wa)
    # Ensure your quad utility is JAX-compatible (e.g., using jax.scipy.integrate or fixed-grid)
    return cg.utils.quad(integrand, 0.0, z, n=20)


def mu(
    z,
    H0=67.9,
    Omega_m=0.307,
    Omega_k=0,
    Omega_r=0,
    w0=-1.0,
    wa=0,
):
    # Map the comoving distance over the array of redshifts
    DCs = jax.vmap(lambda z: comoving_distance(z, H0, Omega_m, Omega_k, Omega_r, w0, wa))(z)
    DH = c_km / H0

    # Transverse comoving distance series for non-flat curvature
    D_TC = 0.0
    for k in range(5):
        D_TC += Omega_k**k * (DCs / DH) ** (2 * k) / factorial(1 + 2 * k)

    # Luminosity Distance
    LD = DCs * D_TC * (1 + z)
    return 5 * jnp.log10(LD) + 25
