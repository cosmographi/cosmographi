import os
import jax.numpy as jnp
from .base import Throughput_wAtmos
import pandas as pd
import numpy as np


class RubinThroughput(Throughput_wAtmos):
    def __init__(self, air_mass=1.2, name=None):
        bands = ["u", "g", "r", "i", "z", "y"]
        w_hardware = []
        T_hardware = []
        loc = os.path.dirname(__file__)
        for b in bands:
            df = pd.read_csv(
                os.path.join(loc, f"transmissions/lsst_hardware_{b}.csv"),
                names=["w", "T"],
                comment="#",
            )
            w_hardware.append(df["w"].values)
            T_hardware.append(df["T"].values)
        w_hardware = np.stack(w_hardware)
        T_hardware = np.stack(T_hardware)
        df = pd.read_csv(
            os.path.join(loc, "transmissions/lsst_atmos_10.csv"), names=["w", "T"], comment="#"
        )
        w_atmosphere = df["w"].values
        T_atmosphere = df["T"].values
        super().__init__(
            bands,
            jnp.array(w_hardware),
            jnp.array(T_hardware),
            jnp.array(w_atmosphere),
            jnp.array(T_atmosphere),
            air_mass=air_mass,
            name=name,
        )
