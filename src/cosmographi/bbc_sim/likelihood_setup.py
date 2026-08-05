
from . import cosmology as cos
import jax.numpy as jnp
def bbc_log_likelihood(z_valid, mu_star_valid, params, C_inv):
   
    Om, w = params
    mu_true = cos.mu(z_valid, Omega_m=Om, w0=w)
    residual = mu_star_valid - mu_true
    return -0.5 * residual @ C_inv @ residual


def log_prior(params, w_lo, w_hi, O_fid, om_sigma):
    Om, w = params

    in_bounds_w  = (w > w_lo) & (w < w_hi)

    diff_om = Om - O_fid
    log_prior_om = -0.5 * (diff_om**2 / om_sigma**2 + jnp.log(2 * jnp.pi * om_sigma**2))
    
    log_prior_w  = -jnp.log(w_hi - w_lo)
    return jnp.where(in_bounds_w, log_prior_om + log_prior_w, -jnp.inf)


def logpost(z_valid, mu_star_valid, params, C_inv, w_lo, w_hi, O_fid, om_sigma):
    return log_prior(params, w_lo, w_hi, O_fid, om_sigma) + bbc_log_likelihood(z_valid, mu_star_valid, params, C_inv)