import numpy as np
import pytest

from osc_physrisk_financial.random_variables import DiscreteRandomVariable

values = [0.1, 0.3, 0.5, 0.7, 0.9]
intervals = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
probabilities = [0.1, 0.3, 0.3, 0.2, 0.1]
percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]

drv = DiscreteRandomVariable(values=values, probabilities=probabilities)
drv2 = DiscreteRandomVariable(intervals=intervals, probabilities=probabilities)
drvs = np.array([drv, 1 / drv])


def test_init():
    assert drv == drv2
    assert drv == DiscreteRandomVariable(values=values, probabilities=probabilities)
    assert (
        np.array_equal(
            np.array([0.1, 0.3, 0.3, 0.2, 0.05]),
            DiscreteRandomVariable(
                intervals=intervals,
                values=None,
                probabilities=[0.1, 0.3, 0.3, 0.2, 0.05],
                convert_to_osc_format=True,
            ).probabilities,
        )
        is False
    )
    assert (
        DiscreteRandomVariable(
            intervals=intervals,
            values=None,
            probabilities=probabilities,
            convert_to_osc_format=True,
        )
        is not None
    )


def test_init_value_errors():
    with pytest.raises(
        ValueError, match="Either intervals or values must be provided."
    ):
        DiscreteRandomVariable(intervals=None, values=None, probabilities=probabilities)

    with pytest.raises(
        ValueError,
        match="Only one of intervals or values should be provided, not both.",
    ):
        DiscreteRandomVariable(
            intervals=intervals, values=values, probabilities=probabilities
        )

    with pytest.raises(ValueError, match="The intervals must be sorted increasingly."):
        DiscreteRandomVariable(
            intervals=[0.4, 0.3, 0.2, 0.1], values=None, probabilities=probabilities
        )

    with pytest.raises(
        ValueError,
        match="The number of intervals must be one more than the number of probabilities.",
    ):
        DiscreteRandomVariable(
            intervals=[0, 0.2, 0.4, 0.6, 0.8], values=None, probabilities=probabilities
        )

    with pytest.raises(
        ValueError, match="The number of values must match the number of probabilities."
    ):
        DiscreteRandomVariable(
            intervals=None, values=[0.1, 0.3, 0.5], probabilities=[0.1, 0.2, 0.3, 0.7]
        )

    with pytest.raises(ValueError, match="The probabilities must sum up to 1."):
        DiscreteRandomVariable(
            intervals=intervals, values=None, probabilities=[0.1, 0.3, 0.3, 0.2, 0.7]
        )

    with pytest.raises(ValueError, match="All probabilities must be between 0 and 1."):
        DiscreteRandomVariable(
            intervals=intervals,
            values=None,
            probabilities=[-0.1, 0.3, 0.3, 0.2, 0.3],
            convert_to_osc_format=True,
        )

    with pytest.raises(
        ValueError, match="Impact bins must be sorted in non-decreasing order."
    ):
        DiscreteRandomVariable(
            intervals=[1.0, 0.8, 0.6, 0.4, 0.2, 0],
            values=None,
            probabilities=probabilities,
            convert_to_osc_format=True,
        )


def test_not_implemented():
    assert NotImplemented == DiscreteRandomVariable(
        intervals=intervals, values=None, probabilities=probabilities
    ).__mul__(other="a")
    assert NotImplemented == DiscreteRandomVariable(
        intervals=intervals, values=None, probabilities=probabilities
    ).__add__(other="a")


def test_rtruediv():
    with pytest.raises(TypeError, match="Numerator must be a real number"):
        DiscreteRandomVariable(
            intervals=intervals, values=None, probabilities=probabilities
        ).__rtruediv__(other="a")

    with pytest.raises(
        ValueError,
        match="Division by zero encountered in DiscreteRandomVariable values",
    ):
        DiscreteRandomVariable(
            intervals=None,
            values=[0.0, 0.3, 0.5, 0.7, 0.9],
            probabilities=probabilities,
        ).__rtruediv__(other=1.0)


def test_eq():
    assert (
        DiscreteRandomVariable(
            intervals=intervals, values=None, probabilities=probabilities
        ).__eq__(1.0)
        is False
    )


def test_check_values():
    assert (
        0.0
        <= DiscreteRandomVariable(
            intervals=intervals, values=None, probabilities=probabilities
        ).check_values(0.0, 1.0)
        <= 1.0
    )


def test_sample():
    assert 4 == len(
        DiscreteRandomVariable(
            intervals=intervals, values=None, probabilities=probabilities
        ).sample(4)
    )


def test_compute_var():
    with pytest.raises(ValueError, match="Percentile must be between 0 and 100."):
        DiscreteRandomVariable(
            intervals=intervals, values=None, probabilities=probabilities
        ).compute_var(percentile=101)


def test_compute_es():
    with pytest.raises(ValueError, match="Percentile must be between 0 and 100."):
        DiscreteRandomVariable(
            intervals=intervals, values=None, probabilities=probabilities
        ).compute_es(percentile=101)


def test_plot():
    DiscreteRandomVariable(
        intervals=intervals, values=None, probabilities=probabilities
    ).plot_pmf()


def test_magic():
    # Negative
    assert -drv == DiscreteRandomVariable(
        values=[-x for x in values], probabilities=probabilities
    )

    # Multiplication
    assert -6 * drv == DiscreteRandomVariable(
        values=[-6 * x for x in values], probabilities=probabilities
    )
    assert -6 * drv == drv * (-6)

    # Addition
    assert 6 + drv == DiscreteRandomVariable(
        values=[6 + x for x in values], probabilities=probabilities
    )
    assert 6 + drv == drv + 6

    # Subtraction
    assert drv - 6 == DiscreteRandomVariable(
        values=[x - 6 for x in values], probabilities=probabilities
    )
    assert 6 - drv == DiscreteRandomVariable(
        values=[6 - x for x in values], probabilities=probabilities
    )

    # Division
    assert -6 / drv == DiscreteRandomVariable(
        values=[-6 / x for x in values], probabilities=probabilities
    )


def test_metrics():
    # Mean
    assert np.isclose(drv.mean(), 0.48)

    # Variance
    assert np.isclose(drv.var(), 0.0516)

    # Exceedance Probability
    assert np.allclose(
        drv.compute_exceedance_probability(),
        np.array([0.9, 0.6, 0.3, 0.1, 0.0]),
    )

    # Occurrence Probability
    assert np.allclose(
        drv.compute_occurrence_probability(1),
        1 - np.exp(np.array([0.1, 0.4, 0.7, 0.9, 1]) - 1),
    )

    # VaR
    assert np.allclose(
        [drv.compute_var(p) for p in percentiles],
        [0.1, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.7, 0.7],
    )

    # Expected Shortfall
    assert np.allclose(
        [drv.compute_es(p) for p in percentiles],
        (48 - np.array([1, 4, 7, 10, 15, 20, 25, 32, 39]))
        / 100
        / (1 - np.array(percentiles) / 100),
    )


def test_metrics_vectorized():
    # Mean
    assert np.allclose(DiscreteRandomVariable.means_vectorized(drvs), [0.48, 2.9968])

    # Variance
    assert np.allclose(
        DiscreteRandomVariable.vars_vectorized(drvs),
        [0.0516, 6.08399],
    )

    # Exceedance Probability
    assert np.allclose(
        DiscreteRandomVariable.compute_exceedance_probability_vectorized(drvs, 0.8),
        [0.1, 1],
    )
    assert np.allclose(
        DiscreteRandomVariable.compute_exceedance_probability_vectorized(drvs, 3),
        [0, 0.4],
    )

    # Occurrence Probability
    assert np.allclose(
        DiscreteRandomVariable.compute_occurrence_probability_vectorized(drvs, 1, 0.8),
        [1 - np.exp(-0.1), 1 - np.exp(-1)],
    )
    assert np.allclose(
        DiscreteRandomVariable.compute_occurrence_probability_vectorized(drvs, 1, 3),
        [0, 1 - np.exp(-0.4)],
    )

    # VaR
    assert np.allclose(
        [DiscreteRandomVariable.compute_var_vectorized(drvs, p) for p in percentiles],
        np.vstack(
            (
                [0.1, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.7, 0.7],
                [10 / 9, 10 / 7, 10 / 7, 2, 2, 2, 10 / 3, 10 / 3, 10 / 3],
            )
        ).transpose(),
    )

    # Expected Shortfall
    assert np.allclose(
        np.array(
            [DiscreteRandomVariable.compute_es_vectorized(drvs, p) for p in percentiles]
        ),
        (
            np.vstack(
                (
                    (48 - np.array([1, 4, 7, 10, 15, 20, 25, 32, 39])) / 100,
                    (1888 - np.array([70, 160, 250, 376, 502, 628, 838, 1048, 1258]))
                    / 630,
                )
            )
            / (1 - np.array(percentiles) / 100)
        ).transpose(),
    )
