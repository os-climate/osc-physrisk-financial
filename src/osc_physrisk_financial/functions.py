"""Auxiliary functions."""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import optimize

# TODO: We should make pypricing library installable so we can import pypricing.pricing.functions.py.
# TODO: Meanwhile we have copied this file in this repository.


def check_all_nonnumeric(arr):
    """Check if all elements in a numpy array or Python list are non-numeric.

    This function tries to convert each element in the array or list to a float. If the
    conversion raises a ValueError or TypeError, or if the value is nan, it means the
    element is non-numeric, so the function continues to the next element. If the
    conversion does not raise an exception and the value is not nan, it means the element
    is numeric, so the function immediately returns False. If the function finishes
    checking all elements without finding a numeric one, it returns True.

    Parameters
    ----------
    arr : numpy.ndarray or list
        The array or list to check.

    Returns
    -------
    bool
        True if all elements are non-numeric, False otherwise.

    """
    for i in arr:
        try:
            val = float(i)
            if not math.isnan(val):
                return False
        except (ValueError, TypeError):
            continue
    return True


def find_root(func, x0, interval, tolerance=10**-8, fprime=None):
    """Find the root of a given function, using several methods.

    Each method is tried in turn until one succeeds.

    If none succeeds, we plot the function in interval.

    Parameters
    ----------
    func : callable
        The function for which the root is to be computed.
    x0 : float
        Initial guess for the root.
    interval : list
        Interval [a,b] for ridder, bisecction and brentq.
    tolerance : float
        If func(solution)>tolerance an exception is raised.
    fprime : callable, optional
        The derivative of the function. If not provided, the Newton method will use the secant method.

    Returns
    -------
    float
        The root found by the successful method (unless all methods failed).

    """
    methods = [
        ("fixed_point", optimize.fixed_point),
        ("newton (Secant)", optimize.newton),
        ("newton (Newton-Raphson)", optimize.newton),
        ("bisection", optimize.bisect),
        ("brentq", optimize.brentq),
        ("ridder", optimize.ridder),
    ]

    root = None

    for name, method in methods:
        try:
            if name == "fixed_point":
                root = method(lambda x: x - func(x), x0)
            elif name == "newton (Secant)":
                root = method(func, x0, fprime=None)
            elif name == "newton (Newton-Raphson)":
                root = method(func, x0, fprime=fprime)
            else:
                root = method(func, interval[0], interval[1])
            # print(f"Method {name} succeeded with root {root}")
            break  # if method succeeded, stop trying the rest
        except Exception:
            pass
    if root is None:
        x_vals = np.linspace(interval[0], interval[1], 200)
        y_vals = [
            func(x) for x in x_vals
        ]  # Note that this code snippet is intentionally not vectorized.
        fig = go.Figure(data=go.Scatter(x=x_vals, y=y_vals))
        fig.update_layout(title="Plot of func", xaxis_title="x", yaxis_title="y")
        fig.show()
        raise Exception("All methods failed")
    else:
        if np.abs(func(root)) > tolerance:  # Maybe another tolerance can be chosen.
            raise Exception("The numerical error is too large.")
        else:
            return root


def dates_formatting(*date_sets):
    """Convert dates to a consistent format and sort them in ascending order.

    Parameters
    ----------
    date_sets : pandas.DatetimeIndex,list of strings, pandas.Timestamp, or string
        Dates to be formatted. It can be a single date (as a pandas.Timestamp or its string representation)
        or an array-like object (as a pandas.DatetimeIndex or a list of its string representation) containing dates.

    Returns
    -------
    sorted_dates : pandas.DatetimeIndex
        A pandas DatetimeIndex object containing the formatted dates in ascending order.

    Examples
    --------
    >>> dates_formatting('2022-01-01')
    DatetimeIndex(['2022-01-01'], dtype='datetime64[ns]', freq=None)

    >>> dates_formatting(['2022-01-03', '2022-01-01', '2022-01-02'])
    DatetimeIndex(['2022-01-01', '2022-01-02', '2022-01-03'], dtype='datetime64[ns]', freq=None)

    >>> dates_formatting(['2022-01-03', '2022-01-01', '2022-01-02'], '2022-01-01')
    [DatetimeIndex(['2022-01-01', '2022-01-02', '2022-01-03'], dtype='datetime64[ns]', freq=None), DatetimeIndex(['2022-01-01'], dtype='datetime64[ns]', freq=None)]

    """
    formatted_dates = []
    for date_set in date_sets:
        if np.asarray(date_set).shape == ():
            dates = [date_set]
        else:
            dates = date_set
        formatted_dates.append(pd.to_datetime(dates).sort_values())
    if len(formatted_dates) == 1:
        formatted_dates = formatted_dates[0]

    return formatted_dates


def contains_word(string_list, word):
    """Check if strings in the given list contain the specified word. Words in each string are separated by underscores.

    Parameters
    ----------
    string_list : list of str
        The list of strings where each string has words separated by underscores.
    word : str
        The word to search for within the strings.

    Returns
    -------
    list of str
        A list of strings from the input `string_list` that contain the specified `word`.

    Examples
    --------
    >>> contains_word(['word1_word2', 'word3_word4', 'word2_word5'], 'word2')
    ['word1_word2', 'word2_word5']

    """
    return [s for s in string_list if word in s.split("_")]
