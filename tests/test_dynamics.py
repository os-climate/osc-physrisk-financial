import numpy as np
import pandas as pd

from osc_physrisk_financial.dynamics import ConstantGrowth

value0 = 1000
growth_rate = 0.05
dates = pd.date_range(start="2020-01-01", periods=5, freq="YE")


def test_init():
    assert (
        ConstantGrowth(growth_rate=growth_rate, value0=value0, name="Test Growth")
        is not None
    )


def test_compute_value():
    const_growth = ConstantGrowth(
        growth_rate=growth_rate, value0=value0, name="Test Growth"
    )
    expected_values = value0 * (1 + growth_rate) ** np.arange(0, 5)
    expected_values = np.array(expected_values)
    assert const_growth.compute_value(dates).all() == expected_values.all()
    assert const_growth.compute_value(dates.tolist()).all() == expected_values.all()
