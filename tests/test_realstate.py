from osc_physrisk_financial.assets import RealAsset
from osc_physrisk_financial.dynamics import ConstantGrowth
from osc_physrisk_financial.random_variables import DiscreteRandomVariable

import pytest
import numpy as np


def test_real_asset():
    # TODO: This script should be transformed in a proper test.

    # Check dynamics
    constant_g = ConstantGrowth(growth_rate=0.02, name="RealAsset", value0=100)
    valuet = constant_g.compute_value(
        dates=["2024-02-09", "2025-12-25", "2023-07-01", "2022-07-01"]
    )
    expected_values = [
        100.0,
        102,
        104.04,
        106.1208,
    ]  # Expected values from simple calculation
    assert np.allclose(valuet, expected_values), f"Value_t = {valuet}"

    # Check Random variables
    values = [0.1, 0.3, 0.5, 0.7, 0.9]
    probabilities = (0.1, 0.2, 0.3, 0.1, 0.3)  # This should sum up to 1
    discrete_rand_var_values = DiscreteRandomVariable(probabilities, values)

    intervals = [0, 0.2, 0.4, 0.6, 0.8, 1]
    probabilities = (0.1, 0.2, 0.3, 0.1, 0.3)  # This should sum up to 1
    discrete_rand_var_intervals = DiscreteRandomVariable(
        probabilities, intervals=intervals
    )
    assert discrete_rand_var_values == discrete_rand_var_intervals
    discrete_rand_var = discrete_rand_var_values
    print(discrete_rand_var.mean())
    discrete_rand_var_1 = 1.3 + discrete_rand_var
    discrete_rand_var_2 = discrete_rand_var + 1.3
    assert discrete_rand_var_1 == discrete_rand_var_2

    five_discrete_rand_var = 5 * discrete_rand_var
    rfive_discrete_rand_var = discrete_rand_var * 5
    assert five_discrete_rand_var == rfive_discrete_rand_var

    divided_rv = 1 / discrete_rand_var

    # Create a numpy array of these random variables
    rv_array = np.array(
        [discrete_rand_var, five_discrete_rand_var, rfive_discrete_rand_var],
        dtype=object,
    )
    # Check __eq__ and np.array stuff
    rv_array_div = 1 / rv_array

    assert divided_rv == rv_array_div[0]  # Dummy test for __rtruediv__ and __eq__

    assert (1 + discrete_rand_var) == (1 + rv_array)[0]  # Dummy test for __sum__


def test_asset():
    constant_g = ConstantGrowth(growth_rate=0.02, name="RealAsset", value0=100)
    probabilities = (0.1, 0.2, 0.3, 0.1, 0.3)  # This should sum up to 1
    values = [0.1, 0.3, 0.5, 0.7, 0.9]
    discrete_rand_var_values = DiscreteRandomVariable(probabilities, values)
    discrete_rand_var = discrete_rand_var_values
    # Check assets
    real_asset = RealAsset(value_0=100, dynamics=constant_g, name="RealState")
    real_asset.financial_losses(["2030-02-09"], damage=discrete_rand_var)

    error_asset = RealAsset(value_0=100, dynamics=None, name="RealState")
    with pytest.raises(ValueError, match="Dynamics must be provided."):
        error_asset.financial_losses(["2030-02-09"], damage=discrete_rand_var)

    # real_asset.financial_losses(["2030-02-09"], damage=discrete_rand_var)[0].plot_pmf()
    losses = real_asset.financial_losses(["2030-02-09"], damage=discrete_rand_var)
    mean_loss = losses[0].mean()
    expected_mean_loss = 56.0
    variance_loss = losses[0].var()
    expected_variance_loss = 724.0
    print(
        f'Mean Financial Losses: {real_asset.financial_losses(["2030-02-09"], damage=discrete_rand_var)[0].mean()}'
    )
    print(
        f'Variance Financial Losses: {real_asset.financial_losses(["2030-02-09"], damage=discrete_rand_var)[0].var()}'
    )
    assert np.allclose(mean_loss, expected_mean_loss), "Mean is not calculated properly"
    assert np.allclose(
        variance_loss, expected_variance_loss
    ), "Variance is not calculated properly"

    intervals_osc = np.array(
        [
            0.00012346,
            0.00021273,
            0.000302,
            0.0003516,
            0.00040436,
            0.00043349,
            0.00048287,
            0.000516,
            0.0005943,
        ]
    )
    probabilities_osc = np.array(
        [
            0.00166667,
            0.00083333,
            0.0005,
            0.00033333,
            0.0002381,
            0.00017857,
            0.00013889,
            0.00011111,
        ]
    )
    discrete_rand_var_osc = DiscreteRandomVariable(
        probabilities=probabilities_osc,
        intervals=intervals_osc,
        convert_to_osc_format=True,
    )

    expected_intervals = np.array(
        [
            0.0,
            0.00012346,
            0.00021273,
            0.000302,
            0.0003516,
            0.00040436,
            0.00043349,
            0.00048287,
            0.000516,
            0.0005943,
        ]
    )
    expected_probabilities = np.array(
        [
            [
                9.96000000e-01,
                1.66666667e-03,
                8.33333333e-04,
                5.00000000e-04,
                3.33333333e-04,
                2.38095238e-04,
                1.78571429e-04,
                1.38888889e-04,
                1.11111111e-04,
            ]
        ]
    )

    assert np.allclose(
        discrete_rand_var_osc.intervals, expected_intervals
    ), "Intervals are not calculated properly"
    assert np.allclose(
        discrete_rand_var_osc.probabilities, expected_probabilities
    ), "Probabilities are not calculated properly"

    # zero included

    intervals_osc_zero = np.array(
        [
            0,
            0.00012346,
            0.00021273,
            0.000302,
            0.0003516,
            0.00040436,
            0.00043349,
            0.00048287,
            0.000516,
        ]
    )
    probabilities_osc_zero = np.array(
        [
            0.00166667,
            0.00083333,
            0.0005,
            0.00033333,
            0.0002381,
            0.00017857,
            0.00013889,
            0.00011111,
        ]
    )

    discrete_rand_var_osc_zero = DiscreteRandomVariable(
        probabilities=probabilities_osc_zero,
        intervals=intervals_osc_zero,
        convert_to_osc_format=True,
    )

    a = np.array(intervals_osc_zero[:-1] + intervals_osc_zero[1:]) / 2
    b = discrete_rand_var_osc_zero.values

    assert np.allclose(a[1:], b[1:]), "Intervals are not calculated properly"
    assert np.isclose(b[0], 0), "Values are not calculated properly"

    # zero not included
    discrete_rand_var_osc_zero = DiscreteRandomVariable(
        probabilities=probabilities_osc,
        intervals=intervals_osc,
        convert_to_osc_format=True,
    )

    a = np.array(intervals_osc[:-1] + intervals_osc[1:]) / 2
    b = discrete_rand_var_osc_zero.values

    assert np.all(np.isclose(a, b[1:])), "Intervals are not calculated properly"
    assert np.isclose(b[0], 0), "Values are not calculated properly"
    # LTV
    damage_1 = 1 / 100 * discrete_rand_var
    damage_2 = 2 / 100 * discrete_rand_var
    damage_3 = 0.01 + 1 / 100 * discrete_rand_var
    loan_amounts = [1, 3, 5]
    damages = [damage_1, damage_2, damage_3]
    ltv = real_asset.ltv(
        dates=["2030-02-09", "2031-02-09"], damages=damages, loan_amounts=loan_amounts
    )

    with pytest.raises(ValueError, match="Dynamics must be provided."):
        error_asset.ltv(
            dates=["2030-02-09", "2031-02-09"],
            damages=damages,
            loan_amounts=loan_amounts,
        )

    with pytest.raises(
        ValueError, match="One or more damages have values outside the 0 to 1 range."
    ):
        damage_4 = damage_1 + 1
        ltv = real_asset.ltv(
            dates=["2030-02-09", "2031-02-09"],
            damages=[damage_4, damage_2, damage_3],
            loan_amounts=loan_amounts,
        )

    with pytest.raises(
        ValueError,
        match="The lengths of 'damage' and 'loan_amount' \\(number of assets\\) must match\\.",
    ):
        ltv = real_asset.ltv(
            dates=["2030-02-09", "2031-02-09"],
            damages=[damage_1, damage_2],
            loan_amounts=loan_amounts,
        )

    print(f" LTV mean value (first date, fist asset): {ltv[0,0].mean()}")
    means = DiscreteRandomVariable.means_vectorized(ltv)
    print(f" LTV mean values: {means}")

    expected_means = np.array(
        [[0.01005639, 0.0303407, 0.05079274], [0.0098592, 0.02974579, 0.0497968]]
    )

    assert np.allclose(means, expected_means), "LTV mean values calculation failed"

    # Variances
    print(f" LTV variance (first date, fist asset): {ltv[0,0].var()}")
    vars = DiscreteRandomVariable.vars_vectorized(ltv)
    print(f" LTV variances: {vars}")

    expected_vars = np.array(
        [
            [7.40214348e-10, 2.72496428e-08, 1.92687839e-08],
            [7.11470923e-10, 2.61915059e-08, 1.85205535e-08],
        ]
    )

    assert np.allclose(vars, expected_vars), "LTV variance calculation failed"

    # VaR
    values = np.array([-100, -20, 0, 50])
    probabilities = np.array([0.1, 0.3, 0.4, 0.2])
    drv_var = DiscreteRandomVariable(values=values, probabilities=probabilities)
    percentile = 95
    # drv_var.plot_pmf()
    var = drv_var.compute_var(percentile=percentile)
    print(f"The Value at Risk (VaR) at the {percentile}% confidence level is: {var}")

    vars = DiscreteRandomVariable.compute_var_vectorized(ltv)
    print(f" LTV VaRs: {vars}")
    print(f"Works as expected? {vars[0][0] == ltv[0][0].compute_var()}")  # Dummy test
    expected_var = 50
    expected_es = 50
    # VaR & ES
    es = drv_var.compute_es(percentile=percentile)
    print(f"Percentile = {percentile}, VaR: {var}, ES: {es}")
    assert np.allclose(var, expected_var), "VaR calculation failed"
    assert np.allclose(es, expected_es), "ES calculation failed"

    # CDF & EP

    values = [0.1, 0.3, 0.5, 0.7, 0.9]
    probabilities = [0.1, 0.3, 0.3, 0.2, 0.1]
    discrete_rand_var = DiscreteRandomVariable(
        values=values, probabilities=probabilities
    )

    _ = discrete_rand_var.compute_cdf()

    check_values = np.linspace(min(values), max(values), 20)
    results = []
    for _ in check_values:
        exceedance_probability = discrete_rand_var.compute_exceedance_probability()
        cdf = discrete_rand_var.compute_cdf()
        sum_check = exceedance_probability + cdf
        results.append(sum_check)

    print(f"Check EP & CDF: {np.allclose(results, 1)}")

    # O(s)

    values = [0.1, 0.3, 0.5, 0.7, 0.9]
    probabilities = [0.1, 0.3, 0.3, 0.2, 0.1]
    lambda_value = 0.5  # Example rate parameter for the Poisson process
    discrete_rand_var = DiscreteRandomVariable(
        values=values, probabilities=probabilities
    )
    _ = discrete_rand_var.compute_occurrence_probability(lambda_value)
