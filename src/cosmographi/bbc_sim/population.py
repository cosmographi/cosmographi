"""
Mock SN Ia population generation: true SALT2 light-curve parameters,
true brightness, and simulated observational scatter.
"""

import jax
import jax.numpy as jnp


def detection_probability(mu_obs, threshold, scale):
    return 1.0 / (1.0 + jnp.exp((mu_obs - threshold) / scale))


def simulate_salt2_population(
    key,
    mu_true,
    z,
    alpha,
    beta,
    sigma_int,
    sigma_obs,
    M0_ref=-19.3,
    M0=-19.3,
    sigma_x1_obs=0.3,
    sigma_c_obs=0.03,
    ref=False,
):
    """
    Simulate
    - a population of supernovae with SALT2 light-curve parameters (x1, c)
    - true and observed peak brightness (mB)
    - the SALT2-reconstructed distance modulus for biascor and true population

    Parameters
    ----------
    key : jax.random.PRNGKey
    mu_true : array, shape (N,)
        True distance modulus for each event.
    z : array, shape (N,)
        Redshifts.
    alpha, beta : float
        SALT2 stretch/color standardization coefficients.
    sigma_int : float
        Intrinsic scatter added to true mB.
    sigma_obs : float
        Observational (measurement) noise on mB.
    M0_ref : float
        Absolute magnitude reference.
    M0 : float
            Absolute magnitude
    sigma_x1_obs, sigma_c_obs : float
        Measurement noise on x1 and c.
    ref : boolean
        BiasCor simulation if True

    Returns
    -------
    dict with keys: z, x1_obs, x1_true, c_obs, c_true, mB_obs, mB_true,
    mu_obs, mu_true.
    """

    N = len(z)
    key, k1, k2, k3, k4, k5, k6 = jax.random.split(key, 7)

    # For BiasCor population
    if ref:
        # Population distributions for stretch (x1) and color (c)
        x1_true = 0.97 + 1.4 * jax.random.normal(k1, (N,))
        c_true = -0.05 + 0.05 * jax.random.normal(k2, (N,))

        # True brightness
        mB_true = (
            mu_true
            - alpha * x1_true
            + beta * c_true
            + M0_ref
            + sigma_int * jax.random.normal(k3, (N,))
        )
    # For true population
    else:
        c_symm = -0.05 + 0.04 * jax.random.normal(k1, (N,))

        skewness = jax.random.exponential(k2, (N,)) * 0.1
        c_true = c_symm + skewness  # asymmetric, red-skewed

        is_young = jax.random.bernoulli(k3, p=0.5, shape=(N,))
        x1_young = 0.6 + 0.9 * jax.random.normal(k4, (N,))
        x1_old = -1.2 + 0.9 * jax.random.normal(k5, (N,))
        x1_true = jnp.where(is_young, x1_young, x1_old)

        # True SALT2 brightness
        mB_true = (
            mu_true - alpha * x1_true + beta * c_true + M0 + sigma_int * jax.random.normal(k6, (N,))
        )

    # Measurement noise
    key, kmB, kx1, kc = jax.random.split(key, 4)
    mB_obs = mB_true + sigma_obs * jax.random.normal(kmB, (N,))
    x1_obs = x1_true + sigma_x1_obs * jax.random.normal(kx1, (N,))
    c_obs = c_true + sigma_c_obs * jax.random.normal(kc, (N,))

    # Observer reconstruction of the distance modulus
    if ref:
        mu_obs = mB_obs + alpha * x1_obs - beta * c_obs - M0_ref
    else:
        mu_obs = mB_obs + alpha * x1_obs - beta * c_obs - M0

    return {
        "z": z,
        "x1_obs": x1_obs,
        "x1_true": x1_true,
        "c_obs": c_obs,
        "c_true": c_true,
        "mB_obs": mB_obs,
        "mB_true": mB_true,
        "mu_obs": mu_obs,
        "mu_true": mu_true,
    }
