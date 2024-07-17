"""Assets definitions."""

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

import osc_physrisk_financial.functions as afsfun
from osc_physrisk_financial.dynamics import Dynamic
from osc_physrisk_financial.random_variables import DiscreteRandomVariable


class Asset(object):
    """Class for instantiating a general Asset.

    Parameters
    ----------
    value_0 : float
        Value of the asset at 0, :math:`V_{0}`

    dynamics : dynamics.Dynamic
        Dynamics assumed for the asset value.

    name : string, optional
        Name for identification.

    cash_flows: Sequence, optional
        Sequence of the associated cash flows (for cash flow generating assets only).

    References
    ----------
    Methodology, Chapter 4 of Methodology survey (Overleaf).

    """

    # TODO: This is not the final parameters list. Check OS-C (assets.py)
    # TODO: we should include latitude: float, longitude: float.

    def __init__(
        self,
        value_0: float,
        dynamics: Optional[Dynamic] = None,
        name: Optional[str] = None,
        cash_flows: Optional[Sequence] = None,
    ):
        """Initialize the AssetClass with dynamics and name.

        `dynamics` or `name` are optional. 'value_0 must be provided'

        Parameters
        ----------
        value_0 : float
            Initial value
        dynamics : Optional[Dynamic] = None
            Asset value dynamics.
        name : Optional[str] = None
            Asset name.
        cash_flows: Optional[Sequence] = None
                Cash flows.

        """
        self.value_0 = value_0
        self.dynamics = dynamics
        self.name = name  # TODO: Not sure if this is useful.
        self.cash_flows = cash_flows

    # TODO: Maybe here we can use OS-C standard:
    # class Asset:
    #     def __init__(self, latitude: float, longitude: float, **kwargs):
    #         self.latitude = latitude
    #         self.longitude = longitude
    #         self.__dict__.update(kwargs)


class RealAsset(Asset):
    """Class for instantiating a Real Asset.

    Parameters
    ----------
    value_0 : float
        Value of the asset at 0, :math:`V_{0}`

    dynamics : dynamics.Dynamic
        Dynamics assumed for the asset value.

    name : string, optional
        Name for identification.

    References
    ----------
    Methodology, Chapter 4 of Methodology survey (Overleaf).

    """

    def __init__(self, value_0: float, dynamics: Dynamic, name: Optional[str] = None):
        """Initialize the RealAssetClass with dynamics and name.

        `dynamics` or `name` are optional. 'value_0 must be provided'

        Parameters
        ----------
        value_0 : float
            Initial value
        dynamics : Optional[Dynamic] = None
            Asset value dynamics.
        name : Optional[str] = None
            Asset name.

        """
        super().__init__(value_0=value_0, dynamics=dynamics, name=None)

    def financial_losses(
        self, dates: Union[pd.DatetimeIndex, list], damage: DiscreteRandomVariable
    ):
        """Compute financial losses for a real asset.

        Parameters
        ----------
        dates :  pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates for which we want to compute :math:`X_t` of [Methodology].  TODO: Do we want to include t_0 here?
        damage : random_variables.RandomVariable
            Damage caused to the asset.

        Returns
        -------
        random_variables.RandomVariable
            Random variable representing  :math:`X_{t}` [Methodology].

        References
        ----------
        Methodology, Chapter 4 of Methodology survey (Overleaf).

        """
        if self.dynamics is None:
            raise ValueError("Dynamics must be provided.")
        dates = afsfun.dates_formatting(dates)
        value_t = self.dynamics.compute_value(dates)
        losses = value_t * damage
        return losses

    def ltv(
        self,
        dates: Union[pd.DatetimeIndex, list],
        damages: Sequence[DiscreteRandomVariable],
        loan_amounts: Sequence[float],
    ):
        r"""Compute Loan To Value (LTV) for a real asset.

        Parameters
        ----------
        dates :  pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates for which we want to compute :math:`X_t` of [Methodology].
            Note that :math: `t_0` should be included here. # TODO: Do we want to include t_0 here?
        damages : Sequence[DiscreteRandomVariable]
            Sequence of DiscreteRandomVariable instances representing damage for each asset.
        loan_amounts : Sequence[float]
            Sequence of floats representing loan amount for each asset.

        Returns
        -------
        random_variables.RandomVariable
            Random variable representing  LTV of [Methodology].
            It returns a numpy.ndarray of 2 dimensions and shape :math:`(\\# dates, \\#  assets)`.

        References
        ----------
        Methodology, Chapter 4 of Methodology survey (Overleaf).

        """
        # Define a function to apply the check to an array of DiscreteRandomVariable instances

        def validate_values(drvs: Sequence[DiscreteRandomVariable]):
            # Vectorize check_values method
            vec_check = np.vectorize(lambda drv: drv.check_values())
            values_valid = vec_check(drvs)

            if not np.all(values_valid):
                raise ValueError(
                    "One or more damages have values outside the 0 to 1 range."
                )

        validate_values(damages)

        if len(damages) != len(loan_amounts):
            raise ValueError(
                "The lengths of 'damage' and 'loan_amount' (number of assets) must match."
            )
        # We reshape for allowing broadcasting
        if self.dynamics is None:
            raise ValueError("Dynamics must be provided.")
        valuet = self.dynamics.compute_value(dates=dates).reshape((len(dates), 1))
        damages_np = np.array(damages)
        damages_mod = (1 + (-1) * damages_np).reshape(
            1, len(damages_np)
        )  # Note that __sub__ is not needed in class DiscreteRandomVariable.
        valuet_sc = valuet * damages_mod
        return loan_amounts / valuet_sc

        # TODO: Maybe it is interesting to vectorize the computation of the mean and variance of the LTVs computed by leveraging numpy.
        #     Check impact_distrib.py from OS-C. As it stands, we can do it using np.vectorize (see methods means_vectorized and means_vectorized)
        #     but we know it is not efficient (essentially a for loop)
        #     https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html#:~:text=returns%20a%20ufunc-,Notes,-The%20vectorize%20function


class PowerPlants(Asset):
    """Class for instantiating a PowerPlant Asset.

    Either `production` or both `capacity` and `av_rate` must be provided. If not directly provided, `production`, is calculated as:
         `production` = `capacity` * `av_rate` * 8760.

    Parameters
    ----------
    dynamics : dynamics.Dynamic
        Dynamics assumed for the asset value.

    name : string, optional
        Name for identification.

    production : float, optional
        Real annual production of a power plant in Wh.

    capacity : float, optional
        Capacity of the power plant in W.

    av_rate : float, optional
        Availability factor of production.

    References
    ----------
    `Canonical_Example_Power_Generation_Plants_Floods` (Overleaf).

    Notes
    -----
    In this case the cash flows are defined through production.

    """

    def __init__(
        self, dynamics: Optional[Dynamic] = None, name: Optional[str] = None, **kwargs
    ):
        """Initialize the PowerPlantsClass with dynamics, name and a variable number of arguments.

        `dynamics` or `name` are optional. 'value_0 must be provided'

        Parameters
        ----------
        dynamics : dynamics.Dynamic
            Dynamics assumed for the asset value.
        name : string, optional
            Name for identification.
        **kwargs : dict
            Variable number of arguments.

        """
        if "production" in kwargs:
            production = kwargs["production"]
            # If not, check if capacity and av_rate are both provided
        elif "capacity" in kwargs and "av_rate" in kwargs:
            production = (
                kwargs["capacity"] * kwargs["av_rate"] * 8760
            )  # Number of hours in a year
        else:
            raise ValueError(
                "Must provide either 'production' or both 'capacity' and 'av_rate'."
            )
        super().__init__(
            value_0=production, dynamics=dynamics, name=None, cash_flows=None
        )

    @staticmethod
    def discount(r: Sequence[float], n: Optional[int] = 1) -> float:
        r"""Compute discount for a given annual evolution of interest rates.

        Parameters
        ----------
        r : Sequence[float]
            An array or sequence including the yearly interest rate for the required period.

        n : int, optional
            By default r is a list containing the yearly interest rates.
            To consider a constant interest rate, introduce the value of the
            interest rate in r and n = number of years to be discounted.

        Returns
        -------
        float
            Float containing the discounting factor calculated as
            :math:`\prod_{i} 1/(1+r_i)^n`.

        """
        if n is not None and n < 1:
            raise ValueError("Discounting cash flows in negative number of year")

        if len(r) > 1 and n != 1:
            raise ValueError("Discounting cash flows has a wrong format")

        aux = np.array(r) + 1
        disc = 1 / np.prod(aux) ** n

        return disc

    def financial_losses(
        self,
        damages: DiscreteRandomVariable,
        energy_price: float,
        r: Sequence[float],
        n: Optional[int] = 1,
    ) -> DiscreteRandomVariable:
        r"""Compute financial losses for a PowerPlant asset.

        Parameters
        ----------
        damages : DiscreteRandomVariable
            Random Variable with the production loss expressed as a decimal (50% :math:`\equiv` 0.5) for each plant.

        energy_price : float
            Average price in €/Wh of the energy production.

        r : list[float]
            An array or sequence containing the annual interest rates.

        n : int, optional
            Number of years to discount.

        Returns
        -------
        DiscreteRandomVariable
            Random Variable containing the financial losses for the asset.

        Notes
        -----
            The use of `r` and `n` follows the same convention as in `discount` method.


        """
        scaled_values = (
            damages.values * self.value_0 * energy_price * self.discount(r, n)
        )
        res = DiscreteRandomVariable(
            values=scaled_values, probabilities=damages.probabilities.tolist()
        )
        return res
