import cosmographi as cp
import jax
import jax.numpy as jnp
import sncosmo
import numpy as np
from scipy.optimize import root
import pytest
from astropy.modeling.physical_models import BlackBody
from astropy import units as u


@pytest.mark.parametrize("T", [1000, 5000, 15000])
def test_blackbody(T, benchmark_jax):
    z = 0.1
    C = cp.Cosmology()
    S = cp.StaticBlackbody(cosmology=C, T=T, R=1, N=1, z=z)
    w = jnp.linspace(100, 2000, 100)  # Wavelengths in nm
    benchmark_jax(S.spectral_flux_density_frequency, cp.utils.flux.nu(w))
    sfd_cp = S.spectral_flux_density_frequency(cp.utils.flux.nu(w))  # in units of erg/s/cm^2/Hz

    # Astropy blackbody
    LD = C.luminosity_distance(z) * cp.utils.constants.Mpc_to_cm  # Convert from Mpc to cm
    bb = BlackBody(temperature=T * u.K)
    luminosity_density = np.pi * (4 * np.pi) * bb(w * u.nm)  # in units of erg/s/Hz
    sfd_astropy = luminosity_density / (4 * jnp.pi * (1 + z) * LD**2)  # in units of erg/s/cm^2/Hz

    assert np.allclose(sfd_cp, sfd_astropy.value, rtol=1e-5, atol=0), (
        "Spectral flux density from StaticBlackbody should match astropy's BlackBody model."
    )


@pytest.mark.parametrize(
    "x0, x1",
    [
        (0.1, 0.5),
        (1.0, -0.5),
        (10, 0.0),
    ],
)
def test_salt2_2021(x0, x1, benchmark):
    c = 0.0
    # Test that the SALT2_2021 source produces a light curve consistent with sncosmo's SALT2 model
    C = cp.Cosmology()
    z = root(lambda z: C.luminosity_distance(z) - 1e-5, 1e-9).x[
        0
    ]  # Find redshift corresponding to 10 pc
    S = cp.SALT2_2021(cosmology=C, z=z, x0=x0, x1=x1, c=c, t0=0.0)
    S.CL.to_static()
    S.M.to_static()
    benchmark(S.load_salt2_model)
    S.load_salt2_model()

    # Get sncosmo's SALT2 model
    source = sncosmo.get_source("salt2", version="T21")
    model = sncosmo.Model(source=source)
    model.set(z=z, t0=0.0, x0=x0, x1=x1, c=c)
    w = jnp.linspace(
        model.minwave() * 1.1, model.maxwave() * 0.9, 500
    )  # Wavelengths in Angstroms (observer frame)

    # Compare loaded models
    assert np.allclose(S.phase_nodes, source._phase), (
        "Phase nodes should match sncosmo's SALT2 model."
    )
    assert np.allclose(S.wavelength_nodes, source._wave / 10), (
        "Wavelength nodes should match sncosmo's SALT2 model."
    )
    assert np.allclose(S.colour_law(w), -source._colorlaw(np.array(w) * 10)), (
        "Colour law should match sncosmo's SALT2 model."
    )
    M0_sncosmo = (
        source._model["M0"](source._phase, source._wave) * 10 / source._SCALE_FACTOR
    )  # convert /Angstrom to /nm
    M0_cp = S.M.value[0] / (4 * np.pi * (10 * 1e-6 * cp.utils.constants.Mpc_to_cm) ** 2)
    assert np.allclose(M0_cp, M0_sncosmo, rtol=1e-10, atol=0), (
        "M0 component of SALT2 model should match sncosmo's implementation."
    )

    # Compare the spectral flux density from our SALT2_2021 implementation to sncosmo's SALT2 model
    t = jnp.linspace(
        model.mintime() * 0.9, model.maxtime() * 0.9, 25
    )  # Time in days (observer frame)
    sfd_sncosmo = (
        np.stack(list(model.flux(t_, w) for t_ in np.array(t))) * 10 / source._SCALE_FACTOR
    )  # convert /Angstrom to /nm

    # Get the spectral flux density from our SALT2_2021 implementation
    sfd_cp = jax.vmap(S.spectral_flux_density, in_axes=(None, 0))(w / 10, t)

    # Note, we use pretty loose atol=0.1 since there are different interpolation schemes used
    # The values are of order 1 so 0.1 just checks gross agreement, not exact agreement
    assert np.allclose(np.array(sfd_cp) / x0, sfd_sncosmo / x0, atol=0.1), (
        "Spectral Flux Density from SALT2_2021 should match sncosmo's SALT2 model."
    )
