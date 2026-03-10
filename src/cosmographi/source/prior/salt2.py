from typing import Literal

import numpy as np

from ...utils import asym_gauss


class SALT2_SK16Prior:
    SK16parameter = {
        "G10": {
            "low-z": {
                "c": (-0.055, 0.023, 0.15),
                "c_std": (0.036, 0.025, 0.034),
                "x1": (0.436, 3.118, 0.724),
                "x1_std": (0.563, 1.582, 0.351),
            },
            "sdss": {
                "c": (-0.038, 0.048, 0.079),
                "c_std": (0.014, 0.009, 0.012),
                "x1": (1.141, 1.653, 0.100),
                "x1_std": (0.032, 0.076, 0.100),
            },
            "ps1": {
                "c": (-0.077, 0.029, 0.121),
                "c_std": (0.023, 0.016, 0.019),
                "x1": (0.604, 1.029, 0.363),
                "x1_std": (0.183, 0.138, 0.121),
            },
            "snls": {
                "c": (-0.065, 0.044, 0.120),
                "c_std": (0.019, 0.013, 0.019),
                "x1": (0.964, 1.232, 0.282),
                "x1_std": (0.136, 0.098, 0.094),
            },
            "high-z": {
                "c": (-0.054, 0.043, 0.101),
                "c_std": (0.011, 0.007, 0.009),
                "x1": (0.973, 1.472, 0.222),
                "x1_std": (0.105, 0.080, 0.076),
            },
            "all": {
                "c": (-0.043, 0.052, 0.107),
                "c_std": (0.010, 0.006, 0.010),
                "x1": (0.945, 1.553, 0.257),
                "x1_std": (0.100, 0.117, 0.078),
            },
        },
        "C11": {
            "low-z": {
                "c": (-0.069, 0.003, 0.148),
                "c_std": (0.008, 0.003, 0.024),
                "x1": (0.419, 3.024, 0.742),
                "x1_std": (0.559, 1.468, 0.347),
            },
            "sdss": {
                "c": (-0.061, 0.023, 0.083),
                "c_std": (0.027, 0.020, 0.018),
                "x1": (1.142, 1.652, 0.104),
                "x1_std": (0.455, 0.232, 0.305),
            },
            "ps1": {
                "c": (-0.103, 0.003, 0.129),
                "c_std": (0.003, 0.003, 0.014),
                "x1": (0.589, 1.026, 0.381),
                "x1_std": (0.179, 0.137, 0.117),
            },
            "snls": {
                "c": (-0.112, 0.003, 0.144),
                "c_std": (0.004, 0.003, 0.017),
                "x1": (0.974, 1.236, 0.283),
                "x1_std": (0.128, 0.094, 0.088),
            },
            "high-z": {
                "c": (-0.099, 0.003, 0.119),
                "c_std": (0.003, 0.003, 0.007),
                "x1": (0.964, 1.467, 0.235),
                "x1_std": (0.105, 0.080, 0.075),
            },
            "all": {
                "c": (-0.062, 0.032, 0.113),
                "c_std": (0.016, 0.011, 0.014),
                "x1": (0.938, 1.551, 0.269),
                "x1_std": (0.101, 0.118, 0.070),
            },
        },
    }

    def __init__(
        self,
        survey: Literal["low-z", "sdss", "ps1", "snls", "high-z", "all"] = "all",
        var: Literal["G10", "C11"] = "G10",
    ):
        self.survey = survey
        self.var = var

    def prior_density(self, x1, c):
        params = self.SK16parameter[self.var][self.survey]
        c_prior = asym_gauss(c, *params["c"])
        x1_prior = asym_gauss(x1, *params["x1"])
        return c_prior * x1_prior

    def prior_sample(self, n):
        params = self.SK16parameter[self.var][self.survey]
        p_low_c = params["c"][1] / (params["c"][1] + params["c"][2])
        p_low_x1 = params["x1"][1] / (params["x1"][1] + params["x1"][2])

        low_mask = np.random.rand(n) < p_low_c
        abs_vals = np.abs(np.random.randn(n))
        c_samples = np.zeros(n)
        c_samples[low_mask] = params["c"][0] - abs_vals[low_mask] * params["c"][1]
        c_samples[~low_mask] = params["c"][0] + abs_vals[~low_mask] * params["c"][2]

        low_mask = np.random.rand(n) < p_low_x1
        abs_vals = np.abs(np.random.randn(n))
        x1_samples = np.zeros(n)
        x1_samples[low_mask] = params["x1"][0] - abs_vals[low_mask] * params["x1"][1]
        x1_samples[~low_mask] = params["x1"][0] + abs_vals[~low_mask] * params["x1"][2]
        return x1_samples, c_samples
