from caskade import Param, forward

from .base import SourceEffect
from ...utils import fp99_extinction_law


class HostExtinction_Fitzpatrick99(SourceEffect):
    """

    Applies the Fitzpatrick 1999 extinction law in the rest frame of the source.

    Citation
    --------
    @ARTICLE{1999PASP..111...63F,
        author = {{Fitzpatrick}, Edward L.},
            title = "{Correcting for the Effects of Interstellar Extinction}",
        journal = {\\pasp},
        keywords = {ISM: DUST, EXTINCTION, Astrophysics},
            year = 1999,
            month = jan,
        volume = {111},
        number = {755},
            pages = {63-75},
            doi = {10.1086/316293},
    archivePrefix = {arXiv},
        eprint = {astro-ph/9809387},
    primaryClass = {astro-ph},
        adsurl = {https://ui.adsabs.harvard.edu/abs/1999PASP..111...63F},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

    """

    def __init__(self, *args, A_V_f99h=None, R_V_f99h=3.1, f99h_active=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.A_V_f99h = Param(
            "A_V_f99h",
            A_V_f99h,
            shape=(),
            description="Fitzpatrick 1999 host dust extinction law scaling",
        )
        self.R_V_f99h = Param(
            "R_V_f99h",
            R_V_f99h,
            shape=(),
            description="Fitzpatrick 1999 host dust extinction law coefficient",
        )
        self.f99h_active = f99h_active

    @forward
    def luminosity_density(self, w, *args, A_V_f99h=None, R_V_f99h=None, **kwargs):
        ld = super().luminosity_density(w, *args, **kwargs)
        if not self.f99h_active:
            return ld
        ext = fp99_extinction_law(w, A_V_f99h, R_V_f99h)
        return ld * 10 ** (-0.4 * ext)


class MWExtinction_Fitzpatrick99(SourceEffect):
    """

    Applies the Fitzpatrick 1999 extinction law in the observer frame to a source.

    Citation
    --------
    @ARTICLE{1999PASP..111...63F,
        author = {{Fitzpatrick}, Edward L.},
            title = "{Correcting for the Effects of Interstellar Extinction}",
        journal = {\\pasp},
        keywords = {ISM: DUST, EXTINCTION, Astrophysics},
            year = 1999,
            month = jan,
        volume = {111},
        number = {755},
            pages = {63-75},
            doi = {10.1086/316293},
    archivePrefix = {arXiv},
        eprint = {astro-ph/9809387},
    primaryClass = {astro-ph},
        adsurl = {https://ui.adsabs.harvard.edu/abs/1999PASP..111...63F},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

    """

    def __init__(self, *args, A_V_f99mw=None, R_V_f99mw=3.1, f99mw_active=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.A_V_f99mw = Param(
            "A_V_f99mw",
            A_V_f99mw,
            shape=(),
            description="Fitzpatrick 1999 host dust extinction law scaling",
        )
        self.R_V_f99mw = Param(
            "R_V_f99mw",
            R_V_f99mw,
            shape=(),
            description="Fitzpatrick 1999 host dust extinction law coefficient",
        )
        self.f99mw_active = f99mw_active

    @forward
    def EBV_to_AV(self, EBV, R_V_f99mw):
        """
        Convert E(B - V) into A_V using standard A_V = R_V * E(B - V) definition.
        """
        return EBV * R_V_f99mw

    @forward
    def spectral_flux_density(self, w, *args, A_V_f99h=None, R_V_f99h=None, **kwargs):
        fd = super().spectral_flux_density(w, *args, **kwargs)
        if not self.f99mw_active:
            return fd
        ext = fp99_extinction_law(w, A_V_f99h, R_V_f99h)
        return fd * 10 ** (-0.4 * ext)
