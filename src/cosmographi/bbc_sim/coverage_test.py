
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from . import cosmology as cos
from . import population as pop
from . import biascor as bc
from jax.scipy.stats import norm as jnorm
from . import ct_likelihood as ct_ll
import blackjax

def sample_priors(n_sims, seed, O_fid, O_sigma, om_lo, om_hi, w_lo, w_hi):

    key = jax.random.PRNGKey(seed)

    key, k1, k2 = jax.random.split(key, 3)

    a, b = (om_lo - O_fid) / O_sigma, (om_hi - O_fid) / O_sigma

    om_true = O_fid + O_sigma * jax.random.truncated_normal(k1, lower=a, upper=b, shape=(n_sims,))

    w_true = jax.random.uniform(k2, shape=(n_sims,), minval=w_lo, maxval=w_hi)

    return jnp.column_stack([om_true, w_true])

def run_sbc(n_sims, n_posterior_samples, lower_z, upper_z, z_mean, z_var,
            alpha, beta, sigma_int_true, sigma_obs,
            sigma_x1_obs, sigma_c_obs,
            cov_matrix, M0,
            threshold, scale, bias_tools, O_fid, O_sigma, om_lo, om_hi, w_lo, w_hi,
            N_events=2000,
            n_chains=4, n_prod_samples=10_000, n_warmup=1000, thin=5,
            max_divergence_frac=0.01,
            seed=0):

    J = jnp.array([1.0, alpha, -beta])
    sigma_mu_stat_i_sq = J @ cov_matrix @ J.T + sigma_int_true
    C_diag_inv = 1.0 / sigma_mu_stat_i_sq  # scalar, same for every sim

    # Stage 1: generate data + bias-correct.
 
    true_params = sample_priors(n_sims=n_sims, seed=seed, O_fid=O_fid, O_sigma=O_sigma,
                                 om_lo=om_lo, om_hi=om_hi, w_lo=w_lo, w_hi=w_hi)
    master_key = jax.random.PRNGKey(seed)
    sim_keys = jax.random.split(master_key, n_sims)

    z_list, mu_star_list, mask_list = [], [], []
    u_om_list, u_w_list = [], []
    warmup_key_list, sample_key_list, ck_list = [], [], []

    for i in range(n_sims):
        om_true, w_true_i = true_params[i]
        key_i = sim_keys[i]
        key_i, kz, kdet, ksim, warmup_key, sample_key, ck = jax.random.split(key_i, 7)

        a = (lower_z - z_mean) / z_var
        b = (upper_z - z_mean) / z_var
        z = z_mean + z_var * jax.random.truncated_normal(kz, lower=a, upper=b, shape=(N_events,))

        mu_true = cos.mu(z=z, w0=w_true_i, Omega_m=om_true)
        sim = pop.simulate_salt2_population(
            ksim, mu_true=mu_true, z=z, alpha=alpha, beta=beta,
            sigma_int=sigma_int_true, sigma_obs=sigma_obs, M0_ref=M0,
            sigma_x1_obs=sigma_x1_obs, sigma_c_obs=sigma_c_obs, M0=M0,
        )

        p_det = pop.detection_probability(sim["mu_obs"], threshold=threshold, scale=scale)
        detected = jax.random.uniform(kdet, shape=p_det.shape) < p_det

        sim_full = {k: sim[k] for k in
                    ("z", "x1_obs", "x1_true", "c_obs", "c_true",
                     "mB_obs", "mB_true", "mu_obs", "mu_true")}

        # concrete arrays here -- not traced, np.asarray works fine
        corr = bc.apply_biascor(sim_full, alpha, beta, bias_tools)
        mu_star = bc.compute_mu_star(corr, alpha, beta, M0)
        full_mask = detected & jnp.asarray(corr["valid"])

        z_list.append(sim_full["z"])
        mu_star_list.append(jnp.asarray(mu_star))
        mask_list.append(full_mask)
        u_om_list.append(ct_ll.to_unconstrained(om_true, om_lo, om_hi))
        u_w_list.append(ct_ll.to_unconstrained(w_true_i, w_lo, w_hi))
        warmup_key_list.append(warmup_key)
        sample_key_list.append(sample_key)
        ck_list.append(ck)

    z_all = jnp.stack(z_list)              # (n_sims, N_events)
    mu_star_all = jnp.stack(mu_star_list)  # (n_sims, N_events)
    mask_all = jnp.stack(mask_list)        # (n_sims, N_events)
    init_pos_all = {"u_om": jnp.stack(u_om_list), "u_w": jnp.stack(u_w_list)}
    warmup_keys_all = jnp.stack(warmup_key_list)
    sample_keys_all = jnp.stack(sample_key_list)
    ck_all = jnp.stack(ck_list)

    # Stage 2: NUTS sampling, vmapped over n_sims  
    def run_one_sim(z, mu_star, full_mask, init_pos, warmup_key, sample_key, ck):
        logpost_fn = lambda u: ct_ll.logpost_unconstrained(
            mu_star, full_mask, C_diag_inv, z, u,
            w_lo, w_hi, O_fid, O_sigma, om_lo, om_hi
        )

        warmup = blackjax.window_adaptation(blackjax.nuts, logpost_fn, target_acceptance_rate=0.8)

        def _do_warmup(k):
            (final_state, params), _ = warmup.run(k, init_pos, num_steps=n_warmup)
            return final_state, params["step_size"], params["inverse_mass_matrix"]

        def run_one_chain(key, init_state, step_size, inverse_mass_matrix):
            kernel = blackjax.nuts(logpost_fn, step_size=step_size,
                                    inverse_mass_matrix=inverse_mass_matrix)
            def body(state, k):
                state, info = kernel.step(k, state)
                return state, (state.position, info.is_divergent)
            keys = jax.random.split(key, n_prod_samples)
            _, (positions, divergences) = jax.lax.scan(body, init_state, keys)
            return positions, divergences

        warmup_keys = jax.random.split(warmup_key, n_chains)
        sample_keys = jax.random.split(sample_key, n_chains)

        init_states, step_sizes, inv_mass = jax.vmap(_do_warmup)(warmup_keys)
        positions, divergences = jax.vmap(run_one_chain)(sample_keys, init_states, step_sizes, inv_mass)

        om_chain = ct_ll.to_constrained(positions["u_om"], om_lo, om_hi)[:, ::thin].reshape(-1)
        w_chain  = ct_ll.to_constrained(positions["u_w"],  w_lo,  w_hi)[:, ::thin].reshape(-1)
        div_flat = divergences[:, ::thin].reshape(-1)

        probs = jnp.where(div_flat, 1e-12, 1.0)
        probs = probs / probs.sum()
        idx = jax.random.choice(ck, om_chain.shape[0], shape=(n_posterior_samples,),
                                    replace=True, p=probs)

        div_frac = jnp.mean(divergences)
        ok = div_frac <= max_divergence_frac

        samples = jnp.where(ok, jnp.stack([om_chain[idx], w_chain[idx]], axis=-1), jnp.nan)
        return samples, div_frac, ok

    s, div_fracs, ok_flags = jax.vmap(run_one_sim)(
        z_all, mu_star_all, mask_all, init_pos_all, warmup_keys_all, sample_keys_all, ck_all
    )

    failed_sims = [{"sim": i, "divergence_frac": float(div_fracs[i])}
                    for i in range(n_sims) if not bool(ok_flags[i])]
    if failed_sims:
        print(f"\n{len(failed_sims)}/{n_sims} sims exceeded divergence threshold:")
        for f in failed_sims:
            print(f"  {f}")

    s = jnp.transpose(s, (1, 0, 2))  # (n_posterior_samples, n_sims, 2)

    # check for NaN anywhere in a sim's samples (any posterior sample, either param)
    sim_has_nan = jnp.any(jnp.isnan(s), axis=(0, 2))   # shape (n_sims,) bool

    n_dropped = int(jnp.sum(sim_has_nan))
    if n_dropped > 0:
        print(f"Dropping {n_dropped}/{s.shape[1]} sims with NaN samples "
            f"(divergence threshold exceeded)")

    keep = ~sim_has_nan
    s_clean = s[:, keep, :]                  # (n_posterior_samples, n_kept_sims, 2)
    true_params_clean = true_params[keep]    # (n_kept_sims, 2)
    return true_params_clean, s_clean, failed_sims