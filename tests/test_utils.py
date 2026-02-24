import jax.numpy as jnp
import jax
import pytest

import cosmographi as cg


@pytest.mark.benchmark
def test_quad_integrate():

    f = lambda x: x**2
    a = 0.0
    b = 1.0
    result = cg.utils.quad(f, a, b, n=50)
    expected = 1.0 / 3.0
    assert jnp.isclose(result, expected, rtol=1e-8)

    log_f = lambda x: jnp.log(x**2)
    log_result = cg.utils.log_quad(log_f, a, b, n=50)
    log_expected = jnp.log(1.0 / 3.0)
    assert jnp.isclose(log_result, log_expected, rtol=1e-8)


def test_gauss_rescale_integrate(benchmark):
    f = lambda x: 1 - (x - 1) ** 2
    a = 0.0
    b = 2.0

    benchmark.pedantic(
        lambda: cg.utils.gauss_rescale_integrate(
            f, a, b, mu=1.0, sigma=0.5, n=50
        ).block_until_ready(),
        warmup_rounds=5,
        rounds=1,
        iterations=50,
    )

    result = cg.utils.gauss_rescale_integrate(f, a, b, mu=1.0, sigma=0.5, n=50)
    expected = 4.0 / 3.0
    assert jnp.isclose(result, expected, rtol=1e-8)

    log_f = lambda x: jnp.log(1 - (x - 1) ** 2)
    log_result = cg.utils.log_gauss_rescale_integrate(log_f, a, b, mu=1.0, sigma=0.5, n=50)
    log_expected = jnp.log(4.0 / 3.0)
    assert jnp.isclose(log_result, log_expected, rtol=1e-8)


def test_integrate_gaussian():
    f = lambda x: jnp.exp(-0.5 * x**2) / jnp.sqrt(2 * jnp.pi)
    a = -10
    b = 10
    res_mid = cg.utils.mid(f, a, b, n=50)
    res_quad = cg.utils.quad(f, a, b, n=50)
    res_gauss = cg.utils.gauss_rescale_integrate(f, a, b, mu=0, sigma=1, n=50)
    expected = 1.0
    assert jnp.isclose(res_mid, expected, rtol=1e-8)
    assert jnp.isclose(res_quad, expected, rtol=1e-8)
    assert jnp.isclose(res_gauss, expected, rtol=1e-8)

    # assert jnp.abs(res_mid - 1.0) > jnp.abs(res_quad - 1.0)
    # assert jnp.abs(res_mid - 1.0) > jnp.abs(res_gauss - 1.0)
    # assert jnp.abs(res_quad - 1.0) > jnp.abs(res_gauss - 1.0)


def test_midpoints():
    mid = cg.utils.midpoints(0, 1, 2)
    expected = jnp.array([0.25, 0.75])
    assert jnp.allclose(mid, expected, rtol=1e-8)

    mid = cg.utils.midpoints(0, 1, 5)
    expected = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
    assert jnp.allclose(mid, expected, rtol=1e-8)


@pytest.mark.parametrize("mu1", [0.0, 1.0, -1.0, 2.0])
@pytest.mark.parametrize("mu2", [-1.0, 0.0, 0.5])
def test_int_Phi_N(mu1, mu2):
    sigma1 = 0.5
    sigma2 = 0.3
    Phi = lambda x: jax.scipy.stats.norm.cdf((x - mu1) / sigma1)
    N = lambda x: jax.scipy.stats.norm.pdf(x, loc=mu2, scale=sigma2)
    res_quad = cg.utils.quad(lambda x: Phi(x) * N(x), -10, 10, 1000)
    res_int = cg.utils.int_Phi_N(mu1, sigma1, mu2, sigma2)
    assert jnp.isclose(res_quad, jnp.exp(res_int), rtol=1e-8)
