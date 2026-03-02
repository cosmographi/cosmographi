import jax
import jax.numpy as jnp
from cosmographi import RubinObservatory, Cosmology, StaticBlackbody


def test_rubin_instrument(benchmark_jax):

    RO = RubinObservatory()
    C = Cosmology()
    BB = StaticBlackbody(cosmology=C, z=1.0, T=5000, R=6e10, N=1e9)

    key = jax.random.PRNGKey(0)
    band = jnp.array(0)
    exp_time = jnp.array(30.0)
    sky_brightness = jnp.array(22.0)
    PSF_Aeff = jnp.array(1.0)
    jroo = jax.jit(RO.observe, static_argnums=(5,))

    F_obs, F_obs_err, F_true, F_true_err = benchmark_jax(
        jroo, key, band, exp_time, sky_brightness, PSF_Aeff, BB
    )
    assert (F_true - F_obs) / F_true_err < 5, "Observed flux should be within 5 sigma of true flux."
    assert (F_true - F_obs) / F_obs_err < 5, "Observed flux should be within 5 sigma of true flux."
