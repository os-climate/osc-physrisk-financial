"""Dynamics."""

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import pandas as pd

import osc_physrisk_financial.functions as afsfun


class Dynamic(ABC):
    """A base class for simulating asset value dynamics.

    Notes
    -----
    This base class is based on Underlying from pypricing.

    """

    def __init__(self, name: Optional[str] = None):
        """Initialize a new instance of Dynamic.

        Attributes
        ----------
        name : string, optional
            Name for identification.

        """
        self.name = name
        self.data = pd.DataFrame()

    @abstractmethod
    def compute_value(self, dates: Union[pd.DatetimeIndex, list]):
        """Abstract method for computing the asset value at future dates.

        Attributes
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Future dates for which the asset value wants to be computed.

        Notes
        -----
            This base class is based on Underlying from pypricing.

        """

    # TODO: Maybe we can use methods like set_data, get_data, get_value, get_dates, get_arithmetic_return, get_return from Underlying.
    # TODO: We have to think about this while developing the code.


class ConstantGrowth(Dynamic):
    r"""Class representing a constant growth model: :math:`V_t = V_0 \\times (1 + \mu)^t.`.

    Parameters
    ----------
    growth_rate : float
        Constant  growth rate :math:`\mu.`

    name : string, optional
        Name for identification.

    value0 : float
        :math:`V_0` in [Methodology]

    Examples
    --------
    >>> cg = ConstantGrowth(growth_rate=0.02, name='RealAsset')

    References
    ----------
    Methodology, Chapter 4 of Methodology survey (Overleaf).

    """

    def __init__(self, growth_rate: float, value0: float, name: Optional[str] = None):
        r"""Initialize a new instance of ConstantGrowth.

        Attributes
        ----------
        growth_rate : float
            Constant  growth rate :math:`\mu.`

        value0 : float
            :math:`V_0` in [Methodology]

        name : string, optional
            Name for identification.

        """
        super().__init__(name=name)
        self.growth_rate = growth_rate
        self.value0 = value0

    def compute_value(self, dates: Union[pd.DatetimeIndex, list]):
        """Compute the asset value at future dates.

        Attributes
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates for which the value wants to be computed. Note that in this model we are only
            interested in the years, so we only extract that part. The initial date is also included
            here ( :math:`t_{0}` such that :math:`V_{t_0} = V_0` of [Methodology].

        Returns
        -------
        np.ndarray
            :math:`V_t` in [Methodology] for the different dates. It includes the value :math:`V_0`.
            Note that the dates have been sorted and the output is returned with the dates sorted.

        References
        ----------
        Methodology, Chapter 4 of Methodology survey (Overleaf).

        """
        dates = afsfun.dates_formatting(dates)
        dates = pd.to_datetime(dates)
        years = dates.year
        years = years - years[0]
        valuet = self.value0 * (1 + self.growth_rate) ** years
        return np.array(valuet)
