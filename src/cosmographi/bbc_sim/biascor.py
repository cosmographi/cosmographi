import jax
from . import cosmology as cos
from . import population as pop
import numpy as np
import functools
from .grid_node import GRID_NODES
from . import lookup as lu


def build_biascor(
    key,
    alpha,
    beta,
    sigma_int,
    sigma_obs,
    sigma_x1_obs,
    sigma_c_obs,
    threshold,
    scale,
    w_fid,
    O_fid,
    n=900_000,
    z_lo=0.02,
    z_hi=1.5,
    M0_ref=-19.3,
    M0=-19.3,
    ref=True,
):
    """
    Simulating SNes with detection cut
    Return: z_det, x1_det, x1_true, c_det, c_true, mb_det, mb_true, mu_det, mu_true
    """
    key, kz = jax.random.split(key, 2)

    # flat z coverage, extending past your data's effective range
    z = jax.random.uniform(kz, shape=(n,), minval=z_lo, maxval=z_hi)
    mu_true = cos.mu(z=z, w0=w_fid, Omega_m=O_fid)

    key, ksim = jax.random.split(key)
    sim = pop.simulate_salt2_population(
        ksim,
        z=z,
        mu_true=mu_true,
        alpha=alpha,
        beta=beta,
        sigma_int=sigma_int,
        sigma_obs=sigma_obs,
        M0_ref=M0_ref,
        M0=M0,
        sigma_x1_obs=sigma_x1_obs,
        sigma_c_obs=sigma_c_obs,
        ref=ref,
    )

    # Detection probability

    p_det = pop.detection_probability(
        sim["mu_obs"],
        threshold,
        scale=scale,
    )

    key, kdet = jax.random.split(key)

    detected = (
        jax.random.uniform(
            kdet,
            shape=p_det.shape,
        )
        < p_det
    )
    return {
        "z": sim["z"][detected],
        "x1_obs": sim["x1_obs"][detected],
        "x1_true": sim["x1_true"][detected],
        "c_obs": sim["c_obs"][detected],
        "c_true": sim["c_true"][detected],
        "mB_obs": sim["mB_obs"][detected],
        "mB_true": sim["mB_true"][detected],
        "mu_obs": sim["mu_obs"][detected],
        "mu_true": sim["mu_true"][detected],
    }


_ALPHA_LO, _ALPHA_HI = GRID_NODES["alpha"][0], GRID_NODES["alpha"][-1]
_BETA_LO, _BETA_HI = GRID_NODES["beta"][0], GRID_NODES["beta"][-1]


def apply_biascor(data, alpha, beta, lookups, min_count=10):
    a = float(np.clip(alpha, _ALPHA_LO, _ALPHA_HI))
    b = float(np.clip(beta, _BETA_LO, _BETA_HI))

    z = np.asarray(data["z"], dtype=float)
    mB = np.asarray(data["mB_obs"], dtype=float)
    x1 = np.asarray(data["x1_obs"], dtype=float)
    c = np.asarray(data["c_obs"], dtype=float)
    N = z.shape[0]

    pts = np.column_stack([z, x1, c, np.full(N, a), np.full(N, b)])

    n_local = lookups["counts"](pts)
    valid = n_local >= min_count

    return {
        "mB_star": mB - lookups["mB"](pts),
        "x1_star": x1 - lookups["x1"](pts),
        "c_star": c - lookups["c"](pts),
        "valid": valid,
        "delta_mB": lookups["mB"](pts),
        "delta_x1": lookups["x1"](pts),
        "delta_c": lookups["c"](pts),
    }


def compute_mu_star(corr, alpha, beta, M0_ref=-19.3):
    return corr["mB_star"] + alpha * corr["x1_star"] - beta * corr["c_star"] - M0_ref


jax.config.update("jax_enable_x64", True)


@functools.lru_cache(maxsize=None)
def build_bias_tools_cached(
    sigma_int,
    sigma_obs,
    sigma_x1_obs,
    sigma_c_obs,
    threshold,
    scale,
    w_fid,
    O_fid,
    valid_counts,
    M0_ref,
    M0,
    n=900_000,
    ref=True,
):
    alpha_nodes = np.array([0.10, 0.18])
    beta_nodes = np.array([2.8, 3.6])

    key = jax.random.PRNGKey(42)
    my_sim_results = []

    for a_val in alpha_nodes:
        a_row = []
        for b_val in beta_nodes:
            key, subkey = jax.random.split(key)
            print(f"Simulating BiasCor at alpha={a_val}, beta={b_val}...")
            bc_node = build_biascor(
                subkey,
                alpha=a_val,
                beta=b_val,
                sigma_int=sigma_int,
                sigma_obs=sigma_obs,
                n=n,
                sigma_x1_obs=sigma_x1_obs,
                sigma_c_obs=sigma_c_obs,
                threshold=threshold,
                scale=scale,
                w_fid=w_fid,
                O_fid=O_fid,
                M0_ref=M0_ref,
                M0=M0,
                ref=ref,
            )
            a_row.append(bc_node)
        my_sim_results.append(a_row)

    bias_tools, _ = lu.build_bbc_lookups(my_sim_results, valid_counts=valid_counts)
    return bias_tools
