import jax.numpy as jnp
import jax
import numpy as np
from tqdm import tqdm
from itertools import combinations_with_replacement


def midpoints(start, end, num):
    """
    Compute midpoints of bins
    """
    rng = (end - start) / num
    return jnp.linspace(start + rng / 2, end - rng / 2, num)


def vmap_chunked1d(func, chunk_size=1024, prog_bar=False, in_axes=0, out_axis=0):
    """
    Vectorized map with chunking to reduce memory usage
    """

    vf = jax.jit(jax.vmap(func, in_axes=in_axes))

    def wrapper(*args):
        local_in_axes = in_axes

        # If in_axes is a single integer (e.g. 0), broadcast it to all args
        if isinstance(local_in_axes, int) or local_in_axes is None:
            local_in_axes = (local_in_axes,) * len(args)
        for ax in range(len(local_in_axes)):
            if local_in_axes[ax] is not None:
                n = args[ax].shape[local_in_axes[ax]]
                break
        results = []
        for i in tqdm(range(0, n, chunk_size), disable=not prog_bar):
            chunk_args = []
            for a, ax in zip(args, local_in_axes):
                if ax is None:
                    chunk_args.append(a)
                else:
                    chunk_args.append(jax.lax.slice_in_dim(a, i, min(i + chunk_size, n), axis=ax))

            results.append(vf(*chunk_args))
        return jnp.concatenate(results, axis=out_axis)

    return wrapper


def int_Phi_N(mu1, sigma1, mu2, sigma2):
    # int_{-inf}^{inf} Phi((x - mu1) / sigma1) N(x|mu2, sigma2^2) dx
    # where Phi is the CDF of the standard normal distribution
    return jax.scipy.stats.norm.logcdf(-(mu1 - mu2) / jnp.sqrt(sigma1**2 + sigma2**2))


@jax.jit
def cdist(x, y):
    return jnp.sqrt(jnp.sum((x[:, None] - y[None, :]) ** 2, -1))


def _polynomial_features(d, degree):
    pows = []
    for deg in range(degree + 1):
        pows.extend(combinations_with_replacement(range(d), deg))
    pows = jnp.array([[i.count(j) for j in range(d)] for i in pows], dtype=jnp.int32)
    return pows


def _polynomial_transform(X, pows):
    return jnp.prod(X[:, None] ** pows[None], axis=-1)


def tdp_regression(X, y, degree):
    """
    Perform polynomial regression with all cross terms up to a given total
    degree.

    First performs X -> X_poly by converting the input features to all
    polynomial features up to the given total degree, including cross terms.
    This means for example in 2 feature dimensions going up to degree d would
    mean all `X0^n * X1^m` where n + m <= d. Then performs linear regression on
    the transformed features to fit the target values y. Uses
    linalg.lstsq(X_poly, y) to solve for the coefficients.

    Parameters: X : array-like, shape (n_samples, n_features)
        The input data.
    y : array-like, shape (n_samples,)
        The target values.
    degree : int
        The total degree of the polynomial features.

    Returns: coeffs : array, shape (n_output_features,)
        The coefficients of the polynomial regression model.
    poly : array, shape (n_output_features, n_features)
        The polynomial feature transformation.
    """

    poly = _polynomial_features(X.shape[1], degree)
    X_poly = _polynomial_transform(X, poly)

    coefs = jnp.linalg.lstsq(X_poly, y)[0]

    return coefs, poly


def tdp_evaluate(X, coefs, poly):
    """
    Evaluate the polynomial regression model at new data points.
    See `tdp_regression` for details.

    Parameters:
    X : array-like, shape (n_samples, n_features)
        The input data.
    coefs : array, shape (n_output_features,)
        The coefficients of the polynomial regression model.
    poly : array, shape (n_output_features, n_features)
        The polynomial feature transformation.

    Returns:
    y_pred : array, shape (n_samples,)
        The predicted values.
    """
    X_poly = _polynomial_transform(X, poly)
    return jnp.dot(X_poly, coefs)


def trim_and_pad_batch(wavelengths, transmissions, threshold=0.01):
    """
    Trims each filter independently, then pads them all to the same
    length (the length of the widest filter).
    """
    # Numpy is faster than jax for this sort of slicing and dicing
    wavelengths, transmissions = np.array(wavelengths), np.array(transmissions)
    n_filters, _ = transmissions.shape

    # 1. Find the start and end indices for every filter
    # mask shape: (n_filters, n_points)
    mask = transmissions > threshold

    starts = []
    ends = []
    widths = []

    for i in range(n_filters):
        idx = np.where(mask[i])[0]
        if idx.size > 0:
            s, e = int(idx[0]), int(idx[-1])
            starts.append(s)
            ends.append(e)
            widths.append(e - s + 1)
        else:
            starts.append(0)
            ends.append(0)
            widths.append(0)

    # 2. Determine the target width (the largest significant region)
    max_w = max(widths)
    # Round for better memory alignment
    max_w = ((max_w + 63) // 64) * 64

    # 3. Create the destination arrays
    # Using NaN for wavelengths is often safer to avoid confusing them with 0.0
    trimmed_trans = np.zeros((n_filters, max_w))
    trimmed_wave = np.full((n_filters, max_w), np.nan)
    dw = np.diff(wavelengths, axis=1).mean(axis=1)  # Average spacing for each filter

    # 4. Populate the arrays
    # We loop because each slice is a different size/location
    for i in range(n_filters):
        if widths[i] > 0:
            s, e = starts[i], ends[i]
            w = widths[i]
            # We place the data at the start of the new array (left-aligned)
            trimmed_trans[i, :w] = transmissions[i, s : e + 1]
            trimmed_wave[i, :w] = wavelengths[i, s : e + 1]

            # Dummy fill the rest of the wavelengths with a linear extrapolation based on the last point
            trimmed_wave[i, w:] = wavelengths[i, e] + dw[i] * np.arange(1, max_w - w + 1)

    return jnp.array(trimmed_wave), jnp.array(trimmed_trans)


def asym_gauss(x, mu, sl, sh):
    """
    Asymmetric Gaussian formed from two Gaussians attached at their peak.

    Parameters
    ----------
    x : jnp.ndarray
        Where to evaluate the asymmetric Gaussian
    mu : jnp.ndarray
        The peak of the asymmetric Gaussian
    sl : jnp.ndarray
        The standard deviation for the lower half of the distribution
    sh : jnp.ndarray
        The standard deviation for the higher half of the distribution
    """

    norm = 1 / ((sl + sh) * jnp.sqrt(jnp.pi / 2))
    return (
        jnp.where(
            x < mu,
            jnp.exp(-0.5 * ((x - mu) / sl) ** 2),
            jnp.exp(-0.5 * ((x - mu) / sh) ** 2),
        )
        * norm
    )
