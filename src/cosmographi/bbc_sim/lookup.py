import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .grid_node import GRID_NODES


def build_bbc_lookups(bc_list, valid_counts):
    edges = [
        np.append(
            GRID_NODES[k] - (GRID_NODES[k][1] - GRID_NODES[k][0]) / 2,
            GRID_NODES[k][-1] + (GRID_NODES[k][1] - GRID_NODES[k][0]) / 2,
        )
        for k in ["z", "x1", "c"]
    ]
    shape = [len(GRID_NODES[k]) for k in ["z", "x1", "c", "alpha", "beta"]]
    sums = {"mB": np.zeros(shape), "x1": np.zeros(shape), "c": np.zeros(shape)}
    counts = np.zeros(shape)

    n_alpha, n_beta = len(GRID_NODES["alpha"]), len(GRID_NODES["beta"])
    for a_idx in range(n_alpha):
        for b_idx in range(n_beta):
            bc = bc_list[a_idx][b_idx]
            sample = np.vstack([bc["z"], bc["x1_obs"], bc["c_obs"]]).T
            for key in ["mB", "x1", "c"]:
                bias = bc[f"{key}_obs"] - bc[f"{key}_true"]
                sums[key][..., a_idx, b_idx], _ = np.histogramdd(sample, bins=edges, weights=bias)
            counts[..., a_idx, b_idx], _ = np.histogramdd(sample, bins=edges)

    lookups = {}
    for key in ["mB", "x1", "c"]:
        grid_mean = np.divide(
            sums[key], counts, out=np.zeros_like(sums[key]), where=counts >= valid_counts
        )
        lookups[key] = RegularGridInterpolator(
            tuple(GRID_NODES.values()),
            grid_mean,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

    # how many events landed in the cell nearest this point
    lookups["counts"] = RegularGridInterpolator(
        tuple(GRID_NODES.values()), counts, method="nearest", bounds_error=False, fill_value=0.0
    )
    return lookups, counts


def get_corrected_params(sn_event, alpha_curr, beta_curr, interp_mB, interp_x1, interp_c):
    point = np.array(
        [[sn_event["z"], sn_event["x1_obs"], sn_event["c_obs"], alpha_curr, beta_curr]], dtype=float
    )

    mB_star = float(sn_event["mB_obs"] - np.squeeze(interp_mB(point)))
    x1_star = float(sn_event["x1_obs"] - np.squeeze(interp_x1(point)))
    c_star = float(sn_event["c_obs"] - np.squeeze(interp_c(point)))

    return mB_star, x1_star, c_star
