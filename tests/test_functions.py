import pytest
import pandas as pd
import numpy as np
from scipy import optimize
from osc_physrisk_financial.functions import (
    find_root,
    check_all_nonnumeric,
    dates_formatting,
    contains_word,
)


# Test for find_root
def func_quad(x):
    return x**2 - 2


def fprime_quad(x):
    return 2 * x


def func_cos(x):
    return np.cos(x) - x


def func_non_quadratic(x):
    return np.tan(x)


def func_cubic(x):
    return x**3 - x - 2


def func_no_real_root(x):
    return x**2 + 1


def func_large_error(x):
    return (x - 1) ** 2 - 0.01  # This will trigger the numerical error too large


interval_quad = [0, 2]
interval_cubic = [1, 2]
interval_no_real_root = [-1, 1]
interval_large_error = [0, 2]


def test_newton_secant_method():
    root = find_root(func_quad, x0=1.0, interval=interval_quad)
    assert np.isclose(root, np.sqrt(2), atol=1e-8)


def test_newton_raphson_method():
    root = find_root(func_quad, x0=1.0, interval=interval_quad, fprime=fprime_quad)
    assert np.isclose(root, np.sqrt(2), atol=1e-8)


def test_fixed_point_method():
    root = find_root(func_non_quadratic, x0=0.5, interval=[0, 1])
    expected_root = optimize.fixed_point(lambda x: x - func_non_quadratic(x), 0.5)
    assert np.isclose(root, expected_root, atol=1e-8)


def test_bisection_method():
    root = find_root(func_cubic, x0=1.5, interval=interval_cubic)
    expected_root = optimize.bisect(func_cubic, interval_cubic[0], interval_cubic[1])
    assert np.isclose(root, expected_root, atol=1e-8)


def test_brentq_method():
    root = find_root(func_cubic, x0=1.5, interval=interval_cubic)
    expected_root = optimize.brentq(func_cubic, interval_cubic[0], interval_cubic[1])
    assert np.isclose(root, expected_root, atol=10**-300)


def test_ridder_method():
    root = find_root(func_cubic, x0=1.5, interval=interval_cubic)
    expected_root = optimize.ridder(func_cubic, interval_cubic[0], interval_cubic[1])
    assert np.isclose(root, expected_root, atol=1e-8)


def test_all_methods_fail():
    with pytest.raises(Exception, match="All methods failed"):
        find_root(func_no_real_root, x0=0, interval=interval_no_real_root)


def test_numerical_error_too_large():
    with pytest.raises(Exception, match="The numerical error is too large."):
        find_root(
            func_large_error, x0=1.0, interval=interval_large_error, tolerance=10**-300
        )


# test for check_all_nonnumeric


def test_all_nonnumeric_empty():
    arr = np.array([])
    assert check_all_nonnumeric(arr)
    arr = []
    assert check_all_nonnumeric(arr)


def test_all_nonnumeric_numeric_elements():
    arr = np.array([1, 2, 3])
    assert not check_all_nonnumeric(arr)
    arr = [1, 2, 3]
    assert not check_all_nonnumeric(arr)


def test_all_nonnumeric_float_and_nan():
    arr = np.array([np.nan, 1.0, 3.5])
    assert not check_all_nonnumeric(arr)
    arr = [np.nan, 1.0, 3.5]
    assert not check_all_nonnumeric(arr)


def test_all_nonnumeric_strings():
    arr = np.array(["abc", "def", "ghi"])
    assert check_all_nonnumeric(arr)
    arr = ["abc", "def", "ghi"]
    assert check_all_nonnumeric(arr)


def test_all_nonnumeric_mixed():
    arr = np.array([1, "abc", np.nan])
    assert not check_all_nonnumeric(arr)
    arr = [1, "abc", np.nan]
    assert not check_all_nonnumeric(arr)


def test_all_nonnumeric_non_iterable():
    with pytest.raises(TypeError):
        check_all_nonnumeric(123)


def test_all_nonnumeric_numeric_with_non_nan():
    arr = np.array([1.0, 2.0, 3.0])
    assert not check_all_nonnumeric(arr)
    arr = [1.0, 2.0, 3.0]
    assert not check_all_nonnumeric(arr)


def test_all_nonnumeric_integers_and_non_nan():
    arr = np.array([1, 2, 3])
    assert not check_all_nonnumeric(arr)
    arr = [1, 2, 3]
    assert not check_all_nonnumeric(arr)


def test_all_nonnumeric_float_and_integer():
    arr = np.array([1.0, 2, 3.5])
    assert not check_all_nonnumeric(arr)
    arr = [1.0, 2, 3.5]
    assert not check_all_nonnumeric(arr)


# test for dates_formatting


def test_single_date_string():
    result = dates_formatting("2022-01-01")
    expected = pd.DatetimeIndex(["2022-01-01"])
    assert result.equals(expected)


def test_list_of_dates_strings():
    result = dates_formatting(["2022-01-03", "2022-01-01", "2022-01-02"])
    expected = pd.DatetimeIndex(["2022-01-01", "2022-01-02", "2022-01-03"])
    assert result.equals(expected)


def test_list_of_dates_strings_with_single_date():
    result = dates_formatting(["2022-01-03", "2022-01-01", "2022-01-02"], "2022-01-01")
    expected1 = pd.DatetimeIndex(["2022-01-01", "2022-01-02", "2022-01-03"])
    expected2 = pd.DatetimeIndex(["2022-01-01"])
    assert result[0].equals(expected1)
    assert result[1].equals(expected2)


def test_pandas_datetime_index():
    dates = pd.to_datetime(["2022-01-03", "2022-01-01", "2022-01-02"])
    result = dates_formatting(dates)
    expected = pd.DatetimeIndex(["2022-01-01", "2022-01-02", "2022-01-03"])
    assert result.equals(expected)


def test_mixed_input_formats():
    dates = ["2022-01-03", "2022-01-01", "2022-01-02"]
    mixed_dates = [pd.to_datetime(dates), "2022-01-01"]
    result = dates_formatting(*mixed_dates)
    expected1 = pd.DatetimeIndex(["2022-01-01", "2022-01-02", "2022-01-03"])
    expected2 = pd.DatetimeIndex(["2022-01-01"])
    assert result[0].equals(expected1)
    assert result[1].equals(expected2)


# test for contains_word (delete if function is deleted from functions.py)


def test_contains_word_single_match():
    string_list = ["word1_word2", "word3_word4", "word2_word5"]
    word = "word2"
    expected_output = ["word1_word2", "word2_word5"]
    assert contains_word(string_list, word) == expected_output
