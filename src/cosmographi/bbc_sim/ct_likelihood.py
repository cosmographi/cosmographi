import jax
import jax.numpy as jnp
from . import cosmology as cos
from jax.scipy.stats import norm as jnorm


def to_unconstrained(x, lo, hi):
    """Map x in (lo, hi) -> u in (-inf, inf)"""
    p = (x - lo) / (hi - lo)
    u = jnp.log(p / (1 - p))
    return u


def to_constrained(u, lo, hi):
    """Map u in (-inf, inf) -> x in (lo, hi)"""
    p = jnp.exp(u) / (1 + jnp.exp(u))
    x = lo + (hi - lo) * p
    return x


def log_abs_jacobian(u, lo, hi):

    log_sig = jax.nn.log_sigmoid(u)  # log(sigmoid(u))
    log_one_minus_sig = jax.nn.log_sigmoid(-u)  # log(1 - sigmoid(u))
    return jnp.log(hi - lo) + log_sig + log_one_minus_sig


def log_likelihood(mu_true, mu_star, full_mask, C_diag_inv):
    """
    mu_star, full_mask: precomputed once per simulation (bias-corrected
    distance mod + valid&detected mask), independent of Om, w.
    mu_obs: depends on Om, w
    C_diag_inv: scalar (1 / sigma_mu_stat_i_sq), diagonal C_stat is uniform.
    """
    residual = jnp.where(full_mask, mu_star - mu_true, 0.0)
    return -0.5 * C_diag_inv * jnp.sum(residual**2)


def log_prior(params, w_lo, w_hi, O_fid, O_sigma, om_lo, om_hi):
    Om, w = params["om"], params["w"]
    in_bounds_w = (w > w_lo) & (w < w_hi)
    in_bounds_om = (Om > om_lo) & (Om < om_hi)

    a = (om_lo - O_fid) / O_sigma
    b = (om_hi - O_fid) / O_sigma
    log_Z = jnp.log(jax.scipy.stats.norm.cdf(b) - jax.scipy.stats.norm.cdf(a))

    # renormalize the density to account for truncation
    log_prior_om = jnorm.logpdf(Om, loc=O_fid, scale=O_sigma) - log_Z

    log_prior_w = -jnp.log(w_hi - w_lo)
    return jnp.where(in_bounds_om & in_bounds_w, log_prior_om + log_prior_w, -jnp.inf)


def logpost(
    mu_true, mu_star, full_mask, C_diag_inv, params, w_lo, w_hi, O_fid, O_sigma, om_lo, om_hi
):
    return log_prior(params, w_lo, w_hi, O_fid, O_sigma, om_lo, om_hi) + log_likelihood(
        mu_true, mu_star, full_mask, C_diag_inv
    )


def logpost_unconstrained(
    mu_star, full_mask, C_diag_inv, z, u, w_lo, w_hi, O_fid, O_sigma, om_lo, om_hi
):
    params = {
        "om": to_constrained(u["u_om"], om_lo, om_hi),
        "w": to_constrained(u["u_w"], w_lo, w_hi),
    }
    log_jac = log_abs_jacobian(u["u_om"], om_lo, om_hi) + log_abs_jacobian(u["u_w"], w_lo, w_hi)

    mu_true = cos.mu(z=z, Omega_m=params["om"], w0=params["w"])
    return (
        log_prior(params, w_lo, w_hi, O_fid, O_sigma, om_lo, om_hi)
        + log_likelihood(mu_true, mu_star, full_mask, C_diag_inv)
        + log_jac
    )
