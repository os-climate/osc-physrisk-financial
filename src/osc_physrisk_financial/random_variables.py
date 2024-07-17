"""functions for random and discrete random variables."""

from abc import ABC, abstractmethod
from typing import Optional, Union, Sequence, Any

import numpy as np
import plotly.graph_objects as go


class RandomVariable(ABC):
    """Abstract class with the common methods and attributes of discrete and continuous random variables.

    Ideally, we wouldn't have to implement this class from scratch, but an initial search seems to indicate
    that what we want doesn't exist in another libraries (like SciPy).
    """

    @abstractmethod
    def __init__(self):
        """Initialize a RandomVariable."""

    @abstractmethod
    def __mul__(self, other: Union[float, int]):
        """Multiply the random variable by a real number. Case  RandomVariable * real number.

        This method scales the pdf or pmf of the random variable by a given scalar
        while keeping the probabilities unchanged.

        Parameters
        ----------
        other : float, or int
            The scalar by which to multiply the pdf or pmf of the random variable.

        Returns
        -------
        RandomVariable
            A new instance of DiscreteRandomVariable with scaled pdf or pmf.

        Notes
        -----
            We define this class since operations like the ones defined are not implemented in scipy.
            For instance: TypeError: unsupported operand type(s) for *: 'int' and 'rv_sample'.

        """

    def __rmul__(self, other: Union[float, int]):
        """Multiply the random variable by a real number. Case real number * RandomVariable.

        This method delegates to `__mul__`, assuming commutativity of the operation.

        Parameters
        ----------
        other : float, or int
            The real number by which to multiply the random variable.

        Returns
        -------
        RandomVariable
            A new instance of DiscreteRandomVariable with scaled pdf or pmf.

        """
        return self.__mul__(other)

    def __neg__(self):
        """Negate the random variable."""
        return self.__mul__(-1)

    @abstractmethod
    def __add__(self, other: Union[float, int]):
        """Add a real number to the random variable. Case RandomVariable + real number.

        This method shifts the pdf or pmf of the random variable by a given number
        while keeping the probabilities unchanged.

        Parameters
        ----------
        other : float, or int
            The real number to add to the pdf or pmf of the random variable.

        Returns
        -------
        RandomVariable
            A new instance of DiscreteRandomVariable with shifted pdf or pmf.

        """

    def __radd__(self, other):
        """Add a real number from the random variable. Case real number + RandomVariable.

        This method is called if the first operand does not support addition
        or returns NotImplemented. It allows commutative addition where the scalar
        is on the left side of the `+`.

        Parameters are the same as __add__.
        """
        # __add__ handles the actual operation, so we just delegate to it.
        return self.__add__(other)

    def __sub__(self, other):
        """Subtract a real number to the random variable. Case RandomVariable - real number.

        __add__ handles the actual operation, so we just delegate to it.

        Parameters are the same as __add__.
        """
        return self.__add__(-other)

    def __rsub__(self, other):
        """Subtract the random variable from a real number. Case real number - RandomVariable.

        __add__ and __mul__ handle the actual operation, so we just delegate to them.

        Parameters are the same as __add__.
        """
        return self.__mul__(-1).__add__(other)

    @abstractmethod
    def __rtruediv__(self, other):
        """Implement division where a real number is divided by a DiscreteRandomVariable.

        Parameters
        ----------
        other : float, or int
            The real number numerator.

        Returns
        -------
            RandomVariable: A new instance representing the result.

        Raises
        ------
            ValueError: If division by any value of the DiscreteRandomVariable is not possible.

        """

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        """Check if the current instance equals another instance of a RandomVariable.

        Parameters
        ----------
        other : Any
            The object to compare against.

        Returns
        -------
        bool
            True if the objects are considered equal, False otherwise.

        """

    @abstractmethod
    def mean(self):
        """Calculate the mean of the random variable.

        Returns
        -------
        float
            The mean of the random variable.

        Notes
        -----
            This is an abstract method and must be implemented by subclasses.

        """

    @staticmethod
    @abstractmethod
    def means_vectorized(rvs: Sequence["RandomVariable"]) -> np.ndarray:
        """Abstract static method to compute means for an array of RandomVariable instances using a vectorized approach.

        Parameters
        ----------
        rvs : Sequence[RandomVariable]
            An array or sequence of RandomVariable instances.

        Returns
        -------
        np.ndarray
            An array of floats representing the means of the random variables.

        Notes
        -----
            This is an abstract method and must be implemented by subclasses.

        """

    @abstractmethod
    def var(self):
        """Calculate the variance of the random variable.

        Returns
        -------
        float
            The variance of the discrete random variable.

        Notes
        -----
            This is an abstract method and must be implemented by subclasses.

        """

    @staticmethod
    @abstractmethod
    def vars_vectorized(rvs: Sequence["RandomVariable"]) -> np.ndarray:
        """Abstract static method to compute variances for an array of RandomVariable instances using a vectorized approach.

        Parameters
        ----------
        rvs : Sequence[RandomVariable]
            An array or sequence of RandomVariable instances.

        Returns
        -------
        np.ndarray
            An array of floats representing the variances of the random variables.

        Notes
        -----
            This is an abstract method and must be implemented by subclasses.

        """

    @abstractmethod
    def compute_cdf(self):
        """Compute the Cumulative Distribution Function (CDF) for the  random variable."""

    @abstractmethod
    def compute_var(self, percentile=95):
        r"""Compute the Value at Risk :math:`V^{p}_{X}` for a random variable :math:`X`.

        The Value at Risk (:math:`V^{p}_{X}`) of a discrete random variable :math:`X` at the level
        :math:`p \in (0, 1)` is the p-quantile of :math:`X` defined by the condition that the cumulative
        distribution function :math:`F_{X}(x)` is greater than or equal to :math:`p`. Formally,
        :math:`V^{p}_{X}` is given by:

        .. math:: V^{p}_{X} := \inf\{x \in \mathbb{R} : P(X \leq x) \geq p\}.

        Notes
        -----
            This is an abstract method and must be implemented by subclasses.

        """

    @staticmethod
    @abstractmethod
    def compute_var_vectorized(rvs):
        """Compute VaRs for an array of RandomVariable instances using a vectorized approach.

        Parameters
        ----------
        rvs : Sequence[RandomVariable]
            An array or sequence of RandomVariable instances.

        Returns
        -------
        np.ndarray
            An array of floats representing the VaRs of the random variables.

        Notes
        -----
            This is an abstract method and must be implemented by subclasses.

        """


class DiscreteRandomVariable(RandomVariable):
    """A class to represent a discrete random variable derived from observed data.

    Parameters
    ----------
    probabilities : array like
        The probabilities associated with each interval or value in the histogram.
    values : array like, optional
        The specific values representing the discrete random variable. Required if `intervals` is not provided.
    intervals : array like, optional
        The intervals (bins) of the histogram representing the discrete random variable. Required if `values` is not provided.
    convert_to_osc_format : bool, optional
        If True, it ensures that the probabilities sum to 1 by adjusting the zero-impact bin.
        This is needed for `ImpactDistrib` from OS-C. Default, False.

    Examples
    --------
    Values Example:

    >>> values = [0.1, 0.3, 0.5, 0.7, 0.9]
    >>> probabilities = [0.1, 0.3, 0.3, 0.2, 0.1]  # This should sum up to 1
    >>> drv = DiscreteRandomVariable(values=values, probabilities=probabilities)

    Intervals Example:

    >>> intervals = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    >>> probabilities = [0.1, 0.3, 0.3, 0.2, 0.1]  # This should sum up to 1
    >>> drv = DiscreteRandomVariable(intervals=intervals, probabilities=probabilities)

    Notes
    -----
        - We use intervals following OS-C convention. Internally, we work with the midpoints of each interval.
        - We define this class since classes like rv_discrete from scipy do not support some important operations like multiplication
          by scalar or adding a scalar to the random variable. However, it would be nice to have these features since they seem standard.
          Maybe from another library outside Scipy.
        - When the probabilities do not sum to one, as in the case of the ImpactDistrib class from OS-C, we add the missing value to zero
          to make the sum equal to one. In this way, we create a "mass point" at zero, meaning that we take the mean value for each interval
          except for zero, where we assign the remaining the probability.
          TODO: We need to check the output (methodology implemented in code) of OS-C impact distribution so we are sure the constructor of
          this class is properly defined. That is to say, verify that methodologically this is what we want given OS-C code.

    """

    def __init__(
        self,
        probabilities: Sequence[Union[float, int]],
        values: Optional[Sequence[Union[float, int]]] = None,
        intervals: Optional[Sequence[Union[float, int]]] = None,
        convert_to_osc_format: Optional[bool] = False,
    ):
        """Initialize the ExampleClass with probabilities, and either values or intervals.

        Exactly one of `values` or `intervals` must be provided.

        Parameters
        ----------
        probabilities : Sequence[Union[float, int]]
            A sequence of probabilities which can be float or int.
        values : Optional[Sequence[Union[float, int]]], optional
            An optional sequence of values corresponding to the probabilities, by default None.
        intervals : Optional[Sequence[Union[float, int]]], optional
            An optional sequence of intervals, by default None.
        convert_to_osc_format : Optional[bool]
            Ensures that the probabilities sum to 1 by adjusting the zero-impact bin. False by default.

        Raises
        ------
        ValueError: If both `values` and `intervals` are provided, or if neither is provided.

        """
        if intervals is None and values is None:
            raise ValueError("Either intervals or values must be provided.")
        if intervals is not None and values is not None:
            raise ValueError(
                "Only one of intervals or values should be provided, not both."
            )

        self.probabilities = np.array(probabilities)
        if intervals is not None:
            probabilities_np = np.array(probabilities)
            intervals_np = np.array(intervals)
            if convert_to_osc_format:
                if not np.all((0 <= probabilities_np) & (probabilities_np <= 1)):
                    raise ValueError("All probabilities must be between 0 and 1.")

                if not np.all(np.diff(intervals_np) >= 0):
                    raise ValueError(
                        "Impact bins must be sorted in non-decreasing order."
                    )
                total_prob = np.sum(probabilities_np)
                print(total_prob)
                if not np.isclose(total_prob, 1):
                    if 0 in intervals_np:
                        zero_index = np.where(intervals_np == 0)[0][0]
                        # Adjust the zero-impact probability
                        probabilities_np[zero_index] += 1 - total_prob
                    else:
                        intervals_np = np.insert(intervals_np, 0, 0)
                        probabilities_np = np.insert(
                            probabilities_np, 0, 1 - total_prob
                        )
                    self.intervals = intervals_np
                    self.probabilities = probabilities_np
                    self.values = (self.intervals[1:-1] + self.intervals[2:]) / 2
                    self.values = np.insert(self.values, 0, 0)
            else:
                self.intervals = intervals_np
                if not (self.intervals == np.sort(self.intervals)).all():
                    raise ValueError("The intervals must be sorted increasingly.")
                if len(self.intervals) != len(probabilities_np) + 1:
                    raise ValueError(
                        "The number of intervals must be one more than the number of probabilities."
                    )
                self.values = (self.intervals[:-1] + self.intervals[1:]) / 2
                self.probabilities = probabilities_np
        else:
            values_np = np.array(values)
            probabilities_np = np.array(probabilities)
            if len(values_np) != len(probabilities_np):
                raise ValueError(
                    "The number of values must match the number of probabilities."
                )
            sorted_indices = np.argsort(values_np)
            self.values = values_np[sorted_indices]
            self.probabilities = probabilities_np[sorted_indices]

        # Ensure probabilities sum up to 1
        if not np.isclose(self.probabilities.sum(), 1):
            raise ValueError("The probabilities must sum up to 1.")

    def __mul__(self, other: Union[float, int]):
        """Multiply the discrete random variable by a scalar.

        This method scales the values of the random variable by a given scalar
        while keeping the probabilities unchanged.

        Parameters
        ----------
        other : float, or int
            The scalar by which to multiply the values of the random variable.

        Returns
        -------
        DiscreteRandomVariable
            A new instance of DiscreteRandomVariable with scaled values.

        """
        if isinstance(other, (int, float)):
            scaled_values = self.values * other
            return DiscreteRandomVariable(
                values=scaled_values, probabilities=self.probabilities.tolist()
            )
        else:
            return NotImplemented

    def __add__(self, other: Union[float, int]):
        """Add a scalar to the discrete random variable.

        This method shifts the values of the random variable by a given scalar
        while keeping the probabilities unchanged.

        Parameters
        ----------
        other : float, or int
            The scalar to add to the values of the random variable.

        Returns
        -------
        DiscreteRandomVariable
            A new instance of DiscreteRandomVariable with shifted values.

        """
        if isinstance(other, (int, float)):
            shifted_values = self.values + other
            return DiscreteRandomVariable(
                values=shifted_values, probabilities=self.probabilities.tolist()
            )
        else:
            return NotImplemented

    def __rtruediv__(self, other: Union[float, int]):
        r"""Implement division where a real number is divided by a DiscreteRandomVariable.

        :math:`a / X` where :math:`a, \\ X` are a Real number and a Discrete Random Variable, respectively.

        Parameters
        ----------
        other : float, or int
            The scalar to add to the values of the random variable.

        Returns
        -------
        DiscreteRandomVariable
            A new instance representing the result.

        Raises
        ------
        ValueError: If division by any value of the DiscreteRandomVariable is not possible.

        Notes
        -----
            We don't really need to define :math:`a / X` but rather :math:`1 / X` since __mul__ and __rmul__
            could be used. For convenience, we have done so, although it wasn't strictly necessary.

        """
        if not isinstance(other, (int, float)):
            raise TypeError("Numerator must be a real number")

        # Check for zeros in self.values to avoid division by zero
        if np.any(self.values == 0):
            raise ValueError(
                "Division by zero encountered in DiscreteRandomVariable values"
            )

        # Calculate new values as the real number divided by each value of the DiscreteRandomVariable
        new_values = other / self.values

        return DiscreteRandomVariable(
            values=new_values, probabilities=self.probabilities.tolist()
        )

    def __eq__(self, other: Any) -> bool:
        """Determine if two DiscreteRandomVariable instances are equal based on their values and probabilities.

        Parameters
        ----------
        other : Any
            The other DiscreteRandomVariable instance to compare against.

        Returns
        -------
        bool
            Returns True if both the values and probabilities match, False otherwise.

        """
        if not isinstance(other, DiscreteRandomVariable):
            return False
        return np.allclose(self.values, other.values) and np.allclose(
            self.probabilities, other.probabilities
        )

    def mean(self):
        """Calculate the mean of the discrete random variable.

        Returns
        -------
        float
            The mean of the discrete random variable.

        Examples
        --------
        >>> values = [0.1, 0.3, 0.5, 0.7, 0.9]
        >>> probabilities = [0.1, 0.3, 0.3, 0.2, 0.1]
        >>> drv = DiscreteRandomVariable(values=values, probabilities=probabilities)
        >>> drv.mean()
        0.48000000000000004

        """
        return np.sum(self.values * self.probabilities)

    @staticmethod
    def means_vectorized(drvs):
        """Compute means for an array of DiscreteRandomVariable instances using a vectorized approach.

        Parameters
        ----------
        drvs : np.ndarray
            An array of DiscreteRandomVariable instances.

        Returns
        -------
        np.ndarray
            An array of floats representing the means of the discrete random variables.

        Notes
        -----
            This method utilizes np.vectorize to apply the mean calculation to each instance in the array. It is primarily
            for convenience and does not offer performance benefits over a traditional loop.

        Examples
        --------
        >>> values = [0.1, 0.3, 0.5, 0.7, 0.9]
        >>> probabilities = [0.1, 0.3, 0.3, 0.2, 0.1]
        >>> drv = DiscreteRandomVariable(values=values, probabilities=probabilities)
        >>> drvs = np.array([drv, 1 / drv])
        >>> DiscreteRandomVariable.means_vectorized(drvs)
        array([0.48     , 2.9968254])

        """
        # TODO: CHeck https://github.com/os-climate/physrisk/blob/main/src/physrisk/kernel/impact_distrib.py#L40
        compute_mean = np.vectorize(lambda drv: drv.mean())
        return compute_mean(drvs)

    def var(self):
        """Calculate the variance of the discrete random variable.

        Returns
        -------
        float
            The variance of the discrete random variable.

        Examples
        --------
        >>> values = [0.1, 0.3, 0.5, 0.7, 0.9]
        >>> probabilities = [0.1, 0.3, 0.3, 0.2, 0.1]
        >>> drv = DiscreteRandomVariable(values=values, probabilities=probabilities)
        >>> drv.var()
        0.05160000000000001

        """
        mean = self.mean()
        variance = np.sum(((self.values - mean) ** 2) * self.probabilities)
        return variance

    @staticmethod
    def vars_vectorized(drvs):
        """Compute variances for an array of DiscreteRandomVariable instances using a vectorized approach.

        Parameters
        ----------
        drvs : np.ndarray
            An array of DiscreteRandomVariable instances.

        Returns
        -------
        np.ndarray
            An array of floats representing the means of the discrete random variables.

        Notes
        -----
            This method utilizes np.vectorize to apply the variance calculation to each instance in the array. It is primarily
            for convenience and does not offer performance benefits over a traditional loop.

        Examples
        --------
        >>> values = [0.1, 0.3, 0.5, 0.7, 0.9]
        >>> probabilities = [0.1, 0.3, 0.3, 0.2, 0.1]
        >>> drv = DiscreteRandomVariable(values=values, probabilities=probabilities)
        >>> drvs = np.array([drv, 1 / drv])
        >>> DiscreteRandomVariable.vars_vectorized(drvs)
        array([0.0516    , 6.08399093])

        """
        compute_var = np.vectorize(lambda drv: drv.var())
        return compute_var(drvs)

    def plot_pmf(self):
        """Plot an interactive histogram representing the probability mass function (PMF) of the discrete random variable.

        This method uses Plotly to create an interactive histogram that provides a visual representation of how
        probabilities are distributed across different intervals.
        """
        # Bar chart with Plotly
        fig = go.Figure(
            data=[
                go.Bar(
                    x=self.values,
                    y=self.probabilities,
                    marker=dict(line=dict(color="black", width=1)),
                )
            ]
        )
        fig.update_layout(
            title="Histogram of the Discrete random variable",
            xaxis_title="Value",
            yaxis_title="Probability",
            bargap=0.2,
        )
        fig.show()

    def check_values(self, min_value: float = 0, max_value: float = 1) -> bool:
        """Check if all values of the DiscreteRandomVariable instance fall within a specified range.

        This method verifies that each value defined in the DiscreteRandomVariable instance is
        between a specified minimum value and maximum value, inclusive. By default, it checks
        whether the values are between 0 and 1.

        Parameters
        ----------
        min_value : float, optional
            The minimum allowable value for the values. This value is inclusive, meaning that
            values can be equal to this minimum value. The default is 0.
        max_value : float, optional
            The maximum allowable value for the values. This value is inclusive, meaning that
            values can be equal to this maximum value. The default is 1.

        Returns
        -------
        bool
            Returns True if all values are within the specified range (min_value to max_value, inclusive).
            Otherwise, returns False.

        Examples
        --------
        >>> values = [0.1, 0.3, 0.5, 0.7, 0.9]
        >>> probabilities = [0.1, 0.3, 0.3, 0.2, 0.1]
        >>> drv = DiscreteRandomVariable(values=values, probabilities=probabilities)
        >>> drv.check_values()
        True
        >>> drv.check_values(0,0.5)
        False

        Notes
        -----
        The method utilizes numpy's vectorized operations to efficiently check all values
        against the provided bounds. This approach is effective for instances with a large
        number of values.

        """
        return bool(np.all((min_value <= self.values) & (self.values <= max_value)))

    def sample(self, n: Optional[int] = 1):
        """Generate `n` random samples from the discrete random variable.

        Parameters
        ----------
        n : int, optional
            The number of samples to generate. The default is 1.

        Returns
        -------
        np.ndarray
            An array of sampled values.

        Examples
        --------
        >>> values = [0.1, 0.3, 0.5, 0.7, 0.9]
        >>> probabilities = [0.1, 0.3, 0.3, 0.2, 0.1]
        >>> drv = DiscreteRandomVariable(values=values, probabilities=probabilities)
        >>> sample = drv.sample(5)

        """
        return np.random.choice(self.values, size=n, p=self.probabilities)

    def compute_cdf(self):
        r"""Compute the Cumulative Distribution Function (CDF) for the discrete random variable.

        The CDF is defined as the probability that the variable takes a value less than or equal to `x`.
        Formally, for a discrete random variable `X` with values `x_i` and corresponding probabilities `p_i`,
        the CDF at a point `x` is given by:

        .. math:: F(x) = P(X \leq x) = \sum_{x_i \leq x} p_i

        Returns
        -------
        cdf : np.ndarray
            An array representing the cumulative probabilities corresponding to the values of the random variable.

        Examples
        --------
        >>> values = [0.1, 0.3, 0.5, 0.7, 0.9]
        >>> probabilities = [0.1, 0.3, 0.3, 0.2, 0.1]
        >>> drv = DiscreteRandomVariable(values=values, probabilities=probabilities)
        >>> drv.compute_cdf()
        array([0.1, 0.4, 0.7, 0.9, 1. ])

        """
        # Compute the cumulative distribution function (CDF)
        cdf = np.cumsum(self.probabilities)

        return cdf

    def compute_exceedance_probability(self):
        """Compute the exceedance probability for a given threshold.

        The exceedance probability is the probability that the discrete random variable exceeds a certain value `x`.
        Formally:

        .. math:: F_X^c(x) = P(X > x) = 1 - F_X(x)

        Returns
        -------
        exceed_prob : np.ndarray
            An array representing the exceedance probabilities corresponding to the values of the random variable.

        Examples
        --------
        >>> values = [0.1, 0.3, 0.5, 0.7, 0.9]
        >>> probabilities = [0.1, 0.3, 0.3, 0.2, 0.1]
        >>> drv = DiscreteRandomVariable(values=values, probabilities=probabilities)
        >>> drv.compute_exceedance_probability()
        array([9.00000000e-01, 6.00000000e-01, 3.00000000e-01, 1.00000000e-01,
               1.11022302e-16])

        """
        cdf = self.compute_cdf()
        exceed_prob = 1 - cdf
        return exceed_prob

    @staticmethod
    def compute_exceedance_probability_vectorized(drvs, x):
        """Compute the exceedance probabilities for an array of DiscreteRandomVariable instances using a vectorized approach.

        Parameters
        ----------
        drvs : np.ndarray
            An array of DiscreteRandomVariable instances.
        x : float
            Value at which to evaluate the exceedance probability function.

        Returns
        -------
        np.ndarray
            An array of floats representing the exceedance probabilities of the discrete random variables evaluated at `x`.

        Notes
        -----
            This method utilizes np.vectorize to apply the exceedance probability calculation to each instance in the array. It is primarily
            for convenience and does not offer performance benefits over a traditional loop.

        Examples
        --------
        >>> values = [0.1, 0.3, 0.5, 0.7, 0.9]
        >>> probabilities = [0.1, 0.3, 0.3, 0.2, 0.1]
        >>> drv = DiscreteRandomVariable(values=values, probabilities=probabilities)
        >>> drvs = np.array([drv, 1 / drv])
        >>> DiscreteRandomVariable.compute_exceedance_probability_vectorized(drvs, 2)
        array([1.11022302e-16, 4.00000000e-01])

        """
        compute_exceedance = np.vectorize(
            lambda drv, x: 1 - np.sum(drv.probabilities[np.where(drv.values <= x)[0]])
        )
        return compute_exceedance(drvs, x)

    def compute_occurrence_probability(self, lambda_value):
        r"""Compute the occurrence probability :math:`O(x)` for the discrete random variable using a Poisson process model.

        We assume i.i.d. random variables.

        In this case we have:

        .. math:: F_X(x) = \\frac{1}{\\lambda} \\log(1 - O(x)) + 1,

        where :math:`F_X(x)` is the CDF of the random variable.

        Parameters
        ----------
        lambda_value : float
            The rate parameter of the Poisson process (number of occurrences per time unit).

        Returns
        -------
        occurrence_prob : np.ndarray
            An array representing the occurrence probabilities O(s) for the values of the random variable.

        Examples
        --------
        >>> values = [0.1, 0.3, 0.5, 0.7, 0.9]
        >>> probabilities = [0.1, 0.3, 0.3, 0.2, 0.1]
        >>> drv = DiscreteRandomVariable(values=values, probabilities=probabilities)
        >>> lambda_value = 0.5  # Example rate parameter for the Poisson process
        >>> drv.compute_occurrence_probability(lambda_value)
        array([0.36237185, 0.25918178, 0.13929202, 0.04877058, 0.        ])

        """
        fs = self.compute_cdf()
        occurrence_prob = 1 - np.exp(-lambda_value * (1 - fs))
        return occurrence_prob

    @staticmethod
    def compute_occurrence_probability_vectorized(drvs, lambda_value, x):
        """Compute the occurrence probabilities at `x` for an array of DiscreteRandomVariable instances using a vectorized approach.

        Parameters
        ----------
        drvs : np.ndarray
            An array of DiscreteRandomVariable instances.
        lambda_value : float
            The rate parameter of the Poisson process (number of occurrences per time unit).
        x : float
            Value at which to evaluate the occurrence probability function.

        Returns
        -------
        np.ndarray
            An array of floats representing the occurrence probabilities of the discrete random variables evaluated at `x`.

        Notes
        -----
            This method utilizes np.vectorize to apply the occurrence probability calculation to each instance in the array. It is primarily
            for convenience and does not offer performance benefits over a traditional loop.

        Examples
        --------
        >>> values = [0.1, 0.3, 0.5, 0.7, 0.9]
        >>> probabilities = [0.1, 0.3, 0.3, 0.2, 0.1]
        >>> drv = DiscreteRandomVariable(values=values, probabilities=probabilities)
        >>> drvs = np.array([drv, 1 / drv])
        >>> lambda_value = 0.5  # Example rate parameter for the Poisson process
        >>> DiscreteRandomVariable.compute_occurrence_probability_vectorized(drvs, lambda_value, 0.3)
        array([0.25918178, 0.39346934])

        """
        compute_occurrence = np.vectorize(
            lambda drv, lambda_value, x: 1
            - np.exp(
                -lambda_value
                * (1 - np.sum(drv.probabilities[np.where(drv.values <= x)[0]]))
            )
        )
        return compute_occurrence(drvs, lambda_value, x)

    def compute_var(self, percentile=95):
        r"""Compute the Value at Risk :math:`V^{p}_{X}` for a discrete random variable :math:`X`.

        The Value at Risk (:math:`V^{p}_{X}`) of a discrete random variable :math:`X` at the level
        :math:`p \in (0, 1)` is the p-quantile of :math:`X` defined by the condition that the cumulative
        distribution function :math:`F_{X}(x)` is greater than or equal to :math:`p`. Formally,
        :math:`V^{p}_{X}` is given by:

        .. math:: V^{p}_{X} := \inf\{x \in \mathbb{R} : P(X \leq x) \geq p\}.

        Parameters
        ----------
        percentile : float, optional
            The confidence level (:math:`p`) for VaR expressed as a percentile (0-100). Default is 95.

        Returns
        -------
        var_value : float
            The computed VaR at the given percentile (confidence level).

        Examples
        --------
        >>> values = [0.1, 0.3, 0.5, 0.7, 0.9]
        >>> probabilities = [0.1, 0.3, 0.3, 0.2, 0.1]
        >>> drv = DiscreteRandomVariable(values=values, probabilities=probabilities)
        >>> drv.compute_var()
        0.9

        """
        if not 0 < percentile < 100:
            raise ValueError("Percentile must be between 0 and 100.")

        # Compute the cumulative distribution function (CDF)
        cdf = self.compute_cdf()

        # Find the index of the first occurrence where the CDF exceeds the target percentile
        # np.isclose is used to avoid comparison numerical errors # TODO: Think of better ways to do this.
        target_index = np.where(
            np.isclose(cdf, percentile / 100.0) + (cdf > percentile / 100.0)
        )[0][0]
        var_value = self.values[target_index]

        return var_value

    @staticmethod
    def compute_var_vectorized(drvs, percentile=95):
        """Compute VaRs for an array of DiscreteRandomVariable instances using a vectorized approach.

        Parameters
        ----------
        drvs : np.ndarray
            An array of DiscreteRandomVariable instances.
        percentile : float, optional
            The confidence level (:math:`p`) for VaR expressed as a percentile (0-100). Default is 95.

        Returns
        -------
        np.ndarray
            An array of floats representing the VaRs of the discrete random variables.

        Notes
        -----
            This method utilizes np.vectorize to apply the VaR calculation to each instance in the array. It is primarily
            for convenience and does not offer performance benefits over a traditional loop.

        Examples
        --------
        >>> values = [0.1, 0.3, 0.5, 0.7, 0.9]
        >>> probabilities = [0.1, 0.3, 0.3, 0.2, 0.1]
        >>> drv = DiscreteRandomVariable(values=values, probabilities=probabilities)
        >>> drvs = np.array([drv, 1 / drv])
        >>> DiscreteRandomVariable.compute_var_vectorized(drvs)
        array([ 0.9, 10. ])

        """
        compute_var_percentile = np.vectorize(
            lambda drv: drv.compute_var(percentile=percentile)
        )
        return compute_var_percentile(drvs)

    def compute_es(self, percentile=95):
        r"""Compute the Expected Shortfall :math:`\\mathrm{ES}^{p}_{X}` for a discrete random variable :math:`X`.

        The Expected Shortfall at level :math:`p` for a discrete random variable :math:`X`, is defined formally as:

        .. math:: \\text{ES}^{p}_X = \\frac{1}{1-p} \int_{p}^{1} V^{q}_X \, dq

        Where :math:`V^{p}_X` is the Value at Risk at level :math:`p`.


        Parameters
        ----------
        percentile : float, optional
            The confidence level (:math:`p`) for ES, expressed as a percentile (0-100). Default is 95.

        Returns
        -------
        es_value : float
            The computed ES at the given percentile (confidence level).

        Raises
        ------
        ValueError
            If `percentile` is not within the range (0, 100).

        Examples
        --------
        >>> values = [0.1, 0.3, 0.5, 0.7, 0.9]
        >>> probabilities = [0.1, 0.3, 0.3, 0.2, 0.1]
        >>> drv = DiscreteRandomVariable(values=values, probabilities=probabilities)
        >>> drv.compute_es()
        0.899999999999998

        """
        # Check that percentile is between 0 and 100
        if not 0 < percentile < 100:
            raise ValueError("Percentile must be between 0 and 100.")

        p = percentile / 100.0
        cdf = self.compute_cdf()

        target_indices = np.where(cdf >= p)[0]

        es = (
            np.sum((self.values * self.probabilities)[target_indices][1:])
            + self.values[target_indices[0]] * (cdf[target_indices[0]] - p)
        ) / (1 - p)

        return es

    @staticmethod
    def compute_es_vectorized(drvs, percentile=95):
        """Compute the Expected Shortfall (ES) for an array of DiscreteRandomVariable instances using a vectorized approach.

        Parameters
        ----------
        drvs : np.ndarray
            An array of DiscreteRandomVariable instances.
        percentile : float, optional
            The confidence level (:math:`p`) for ES expressed as a percentile (0-100). Default is 95.

        Returns
        -------
        np.ndarray
            An array of floats representing the ESs of the discrete random variables.

        Notes
        -----
            This method utilizes np.vectorize to apply the ES calculation to each instance in the array. It is primarily
            for convenience and does not offer performance benefits over a traditional loop.

        Examples
        --------
        >>> values = [0.1, 0.3, 0.5, 0.7, 0.9]
        >>> probabilities = [0.1, 0.3, 0.3, 0.2, 0.1]
        >>> drv = DiscreteRandomVariable(values=values, probabilities=probabilities)
        >>> drvs = np.array([drv, 1 / drv])
        >>> DiscreteRandomVariable.compute_es_vectorized(drvs)
        array([ 0.9, 10. ])

        """
        compute_es_percentile = np.vectorize(
            lambda drv: drv.compute_es(percentile=percentile)
        )
        return compute_es_percentile(drvs)
