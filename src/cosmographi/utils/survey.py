from scipy.spatial import KDTree
import numpy as np


def radec_to_sphere(ra, dec):
    ra_rad = np.radians(ra + 180)
    dec_rad = np.radians(dec + 90)
    x = np.sin(dec_rad) * np.cos(ra_rad)
    y = np.sin(dec_rad) * np.sin(ra_rad)
    z = np.cos(dec_rad)
    return np.stack((x, y, z), axis=1)  # (N, 3)


def cross_match_survey_circle(
    source_tmin,
    source_tmax,
    source_ra,
    source_dec,
    survey_t,
    survey_ra,
    survey_dec,
    survey_fov,
    chunk_time=6 * 30,
):
    """
    Cross match between a survey and a list of transients.

    Search through the times (survey_t) and positions (survey_ra, survey_dec) of
    a survey imaging campaign and check which ones would overlap a transient
    source. The transient is assumed to be a point source (at source_ra,
    source_dec), and only visible between source_tmin and source_tmax. The
    observable region is assumed to be circular with diameter survey_fov.

    This may be sufficient for simulation purposes, or may just be a good
    initial trimming of a massive survey and transient database into a more
    manageable size.

    The result is a dict mapping each source index to a 1D array of indices
    of the corresponding observations in the survey.

    Parameters
    ----------
    source_tmin : np.array
        Start time for the transient, earliest time it is visible. (MJD)
    source_tmax : np.array
        End time for the transient, last time it is visible. (MJD)
    source_ra : np.array
        Right Ascension coordinate for transient. (deg)
    source_dec : np.array
        Declination coordinate for transient. (deg)
    survey_t : np.array
        Image time, assumed essentially instantaneous. (MJD)
    survey_ra : np.array
        Right Ascension coordinate of center of image. (deg)
    survey_dec : np.array
        Declination coordinate of center of image. (deg)
    survey_fov : float
        Diameter of circular field of view of survey images. (deg)
    chunk_time : float
        The sources and survey are chunked into blocks of this much time.
        Trimming on a single axis such as time is very fast and can massively
        reduce the number of cross matches to make, speeding up the whole
        process if chosen correctly. The default chunk_time is chosen for a
        source transient that has a timescale on the order of a few months.

    Returns
    -------
    match_indices : dict[list]
        Matches between sources and survey images. Dict with entry for every
        source that has one or more matches (identified by index in original
        array), with elements that are lists of the indices of matching survey
        observations.
    """

    Nsrc = np.arange(len(source_tmin))
    Nsrv = np.arange(len(survey_t))

    coord_src = radec_to_sphere(source_ra, source_dec)
    coord_srv = radec_to_sphere(survey_ra, survey_dec)
    # Get distance between two points separated by the fov radius
    coord_r = np.linalg.norm(
        np.diff(radec_to_sphere(np.zeros(2), np.array([0, survey_fov / 2])), axis=0)
    )

    t_range = np.max(survey_t) - np.min(survey_t)
    n_chunks = max(2, int(np.ceil(t_range / chunk_time)) + 1)
    t_chunks = np.linspace(np.min(survey_t), np.max(survey_t) + chunk_time / 100, n_chunks)

    all_matches = {}
    for tstart, tend in zip(t_chunks[:-1], t_chunks[1:]):
        sel_src = (source_tmax >= tstart) & (source_tmin < tend)  # src with any overlap in window
        sel_srv = (survey_t >= tstart) & (survey_t < tend)  # imgs in window

        # Skip this time chunk if there are no selected sources or survey pointings.
        if not np.any(sel_src) or not np.any(sel_srv):
            continue

        tree_src = KDTree(coord_src[sel_src])
        tree_srv = KDTree(coord_srv[sel_srv])
        matches = tree_src.query_ball_tree(tree_srv, coord_r)

        Nsrc_chunk = Nsrc[sel_src]
        Nsrv_chunk = Nsrv[sel_srv]

        src_tmin_chunk = source_tmin[sel_src]
        src_tmax_chunk = source_tmax[sel_src]
        srv_t_chunk = survey_t[sel_srv]

        for i, m in enumerate(matches):
            if len(m) == 0:
                continue
            # Further refine selection by exact time, rather than chunk membership
            m = np.array(m, dtype=int)
            m = m[(srv_t_chunk[m] > src_tmin_chunk[i]) & (srv_t_chunk[m] < src_tmax_chunk[i])]
            if len(m) == 0:
                continue
            isrc = int(Nsrc_chunk[i])
            if isrc in all_matches:
                all_matches[isrc].append(Nsrv_chunk[m])
            else:
                all_matches[isrc] = [Nsrv_chunk[m]]

    for key in all_matches:
        all_matches[key] = np.concatenate(all_matches[key])

    return all_matches
