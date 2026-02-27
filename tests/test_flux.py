import pytest
import jax.numpy as jnp

from cosmographi.throughput import RubinThroughput
from cosmographi.utils import flux


def test_ABMag_consistency(benchmark):
    f = 3631 * 1e-23  # Convert Jy to erg/s/cm^2/Hz

    benchmark(RubinThroughput)

    throughput = RubinThroughput()

    # photons/s/cm^2 in each band computed using f_nu and f_lambda should be consistent
    for i in range(len(throughput.bands)):
        bandflux_nu = flux.f_nu_band(throughput.nu[i], f, throughput.T_nu(throughput.nu[i], i))
        bandflux_w = flux.f_lambda_band(
            throughput.w[i], flux.f_l(throughput.nu[i], f)[::-1], throughput.T(throughput.w[i], i)
        )

        assert jnp.allclose(bandflux_nu, bandflux_w), (
            "Fluxes computed in f_nu and f_lambda should be consistent."
        )


@pytest.mark.benchmark
def test_f_round_trip():
    # Test that converting from f_nu to f_lambda and back gives the original value
    w = jnp.linspace(200, 1000, 800)  # Wavelengths in nm
    f_nu = 3631 * 1e-23  # Flux density in erg/s/cm^2/Hz

    nu = flux.nu(w)  # Frequencies corresponding to the wavelengths
    f_lambda = flux.f_l(nu, f_nu)
    f_nu_round_trip = flux.f_nu(w, f_lambda)

    assert jnp.allclose(f_nu, f_nu_round_trip), (
        "Converting from f_nu to f_lambda and back should give the original value."
    )


@pytest.mark.benchmark
def test_redshift_conversions():
    # Test that converting from rest to observer frame and back gives the original value
    w_rest = jnp.array([400, 500, 600])  # Rest frame wavelengths in nm
    t_rest = jnp.array([1, 10, 20])  # Rest frame times in days
    z = 0.5

    w_obs = flux.rest_to_observer_wavelength(w_rest, z)
    t_obs = flux.rest_to_observer_time(t_rest, z)

    assert jnp.all(w_rest < w_obs), (
        "Observer frame wavelengths should be longer than rest frame wavelengths."
    )
    assert jnp.all(t_rest < t_obs), "Observer frame times should be longer than rest frame times."

    w_rest_round_trip = flux.observer_to_rest_wavelength(w_obs, z)
    t_rest_round_trip = flux.observer_to_rest_time(t_obs, z)

    assert jnp.allclose(w_rest, w_rest_round_trip), (
        "Converting from rest to observer wavelength and back should give the original value."
    )
    assert jnp.allclose(t_rest, t_rest_round_trip), (
        "Converting from rest to observer time and back should give the original value."
    )


def test_f_lambda_redshift(benchmark_jax):
    # Test that f_lambda correctly accounts for redshift effects
    z = 0.5
    DL = 1000  # Luminosity distance in Mpc
    LD = jnp.array([1e40, 1e41, 1e42])  # Luminosity density in erg/s/nm

    benchmark_jax(flux.f_lambda, z, DL, LD)

    f_obs = flux.f_lambda(z, DL, LD)

    # The observed flux should be lower than the luminosity density due to distance and redshift effects
    assert jnp.all(f_obs < LD), (
        "Observed flux should be lower than luminosity density due to redshift and distance effects."
    )

    f_obs_expected = jnp.array([5.57182909e-17, 5.57182909e-16, 5.57182909e-15])
    assert jnp.allclose(f_obs, f_obs_expected), (
        "Observed flux should match the expected value for the given redshift, luminosity distance, and luminosity density."
    )
