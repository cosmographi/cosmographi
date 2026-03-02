import jax.numpy as jnp
import jax
import numpy as np
from tqdm import tqdm
from scipy.stats.qmc import LatinHypercube


def mala(
    initial_state,  # (num_chains, D)
    log_prob,  # f(x) -> logP ()
    num_samples,
    epsilon,
    mass_matrix=None,  # covariance
    progress=True,
    desc="MALA",
    key=None,
):
    x = jnp.array(initial_state, copy=True)
    C, D = x.shape
    vlogP = jax.jit(jax.vmap(log_prob))
    vlogP_grad = jax.jit(jax.vmap(jax.grad(log_prob)))

    # mass, inv_mass, L
    if mass_matrix is None:
        mass = jnp.eye(D)  # (D, D)
    else:
        mass = jnp.array(mass_matrix, copy=True)  # (D, D)
    inv_mass = jnp.linalg.inv(mass)  # (D, D)
    L = jnp.linalg.cholesky(mass)  # (D, D)

    samples = jnp.zeros((num_samples, C, D), dtype=x.dtype)  # (N, C, D)
    accept = jnp.zeros((num_samples, C), dtype=bool)  # (N,C)

    # Cache current state
    logp_cur = vlogP(x)  # (C,)
    grad_cur = vlogP_grad(x)  # (C, D)

    # Random number generator
    if key is None:
        key = jax.random.PRNGKey(np.random.randint(1e10))

    it = range(num_samples)
    if progress:
        it = tqdm(it, desc=desc, position=0, leave=True)

    for t in it:
        # proposal using current grad
        mu_x = 0.5 * (epsilon**2) * (grad_cur @ mass)  # (C, D)
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, (C, D)) @ L.T  # (C, D)
        x_prop = x + mu_x + epsilon * noise  # (C, D)

        # Evaluate proposal
        logp_prop = vlogP(x_prop)  # (C,)
        grad_prop = vlogP_grad(x_prop)  # (C, D)

        mu_xprop = 0.5 * (epsilon**2) * (grad_prop @ mass)  # (C, D)

        # q(x|x') \propto \exp(-0.5|x - x' - mu(x')|^2 / \epsilon^2)
        d1 = x - x_prop - mu_xprop  # for q(x | x')
        d2 = x_prop - x - mu_x  # for q(x'| x)

        logq1 = -0.5 * jnp.einsum("bi,ij,bj->b", d1, inv_mass, d1) / epsilon**2  # (C,)
        logq2 = -0.5 * jnp.einsum("bi,ij,bj->b", d2, inv_mass, d2) / epsilon**2  # (C,)

        log_alpha = (logp_prop - logp_cur) + (logq1 - logq2)  # (C,)

        # Accept or reject
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, (C,))  # (C,)
        accept = accept.at[t].set(jnp.log(u) < log_alpha)  # (N, C)

        # Update all three pieces in-place where accepted
        x = x.at[accept[t]].set(x_prop[accept[t]])  # (C, D)
        logp_cur = logp_cur.at[accept[t]].set(logp_prop[accept[t]])  # (C,)
        grad_cur = grad_cur.at[accept[t]].set(grad_prop[accept[t]])  # (C, D)

        samples = samples.at[t].set(x)  # (N, C, D)

        if progress:
            it.set_postfix(acc_rate=f"{accept[: t + 1].mean():0.2f}")

    return samples


def latin_hypercube(m, n, d, bounds=None, seed=None):
    """
    Run scipy Latin Hypercube sampling m times for samples of n points in d dimensions.

    Args:
        m: number of random sets to draw
        n: number of points in a random set
        d: dimensionality of the hypercube
        bounds (optional): length (2,d) np.array giving (min, max) bounds of hyper-cube
        seed: int to pass to np.random.default_rng to make random number generator
    """

    rng = np.random.default_rng(seed)
    sampler = LatinHypercube(d=d, rng=rng)
    lhs = list(sampler.random(n=n) for _ in range(m))
    lhs = np.stack(lhs)

    if bounds is not None:
        lhs = lhs * (bounds[1] - bounds[0]) + bounds[0]
    return jnp.array(lhs)


def asym_gauss(x, mu, sigma_left, sigma_right):
    """
    Asymmetric Gaussian formed from two Gaussians attached at their peak.

    Parameters
    ----------
    x : jnp.ndarray
        Where to evaluate the asymmetric Gaussian
    mu : jnp.ndarray
        The peak of the asymmetric Gaussian
    sigma_left : jnp.ndarray
        The standard deviation for the lower half of the distribution
    sigma_right : jnp.ndarray
        The standard deviation for the higher half of the distribution
    """

    norm = 1 / ((sigma_left + sigma_right) * jnp.sqrt(jnp.pi / 2))
    return (
        jnp.where(
            x < mu,
            jnp.exp(-0.5 * ((x - mu) / sigma_left) ** 2),
            jnp.exp(-0.5 * ((x - mu) / sigma_right) ** 2),
        )
        * norm
    )


def sample_asym_gauss(mu, sigma_left, sigma_right, n):
    """
    Sample from an asymmetric Gaussian distribution with mean `mu`, left std `sigma_left`, and right std `sigma_right`.
    """

    p_low = sigma_left / (sigma_left + sigma_right)
    low_mask = np.random.rand(n) < p_low
    abs_vals = np.abs(np.random.randn(n))
    samples = np.zeros(n)
    samples[low_mask] = mu - abs_vals[low_mask] * sigma_left
    samples[~low_mask] = mu + abs_vals[~low_mask] * sigma_right
    return np.array(samples)
