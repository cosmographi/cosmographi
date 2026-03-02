from caskade import Module
from ..throughput import Throughput


class BaseSurvey(Module):
    """
    Base class for representing astronomical surveys.

    The survey is composed of a table of observations, each with attributes such
    as right ascension (RA), declination (Dec), time of observation, and
    observing conditions. Each observation should also be represented by a
    unique key. This way one can query the survey for matching observations to
    get a set of keys, then use those keys to retrieve the relevant observation
    attributes.
    """

    def __init__(self, filters: Throughput, name=None):
        super().__init__(name)
        self.filters = filters


class TimeDomainSurvey(BaseSurvey):
    """
    Class for representing time-domain astronomical surveys.

    This class extends the BaseSurvey to include methods specific to time-domain
    observations, such as sampling observation times.
    """

    def __init__(self, filters: Throughput, name=None):
        super().__init__(filters, name)
