import numpy as np

# Nodes are the centers of the cells
GRID_NODES = {
    "z": np.arange(0.025, 2.15, 0.05),  # dz=0.05
    "x1": np.arange(-3.25, 5.5, 0.5),
    "c": np.arange(-0.225, 0.30, 0.05),
    "alpha": np.array([0.10, 0.18]),
    "beta": np.array([2.8, 3.6]),
}
