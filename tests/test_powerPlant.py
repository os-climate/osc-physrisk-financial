import numpy as np
import pytest

from osc_physrisk_financial.assets import PowerPlants
from osc_physrisk_financial.random_variables import DiscreteRandomVariable


def test_power_plants():
    # Check Random variables
    values = [0.1, 0.3, 0.5, 0.7, 0.9]
    probabilities = (0.1, 0.2, 0.3, 0.1, 0.3)  # This should sum up to 1
    _ = DiscreteRandomVariable(probabilities, values)

    intervals = [0, 0.2, 0.4, 0.6, 0.8, 1]
    probabilities = (0.1, 0.2, 0.3, 0.1, 0.3)  # This should sum up to 1
    drv_intervals = DiscreteRandomVariable(probabilities, intervals=intervals)

    prod = 7892 * (10**9)  # Wh generated in 2019
    elec_price = 48.87 / (10**6)  # euros/Wh
    name = "Central Nuclear Trillo"

    with pytest.raises(
        ValueError,
        match="Must provide either 'production' or both 'capacity' and 'av_rate'.",
    ):
        PowerPlants()

    pp = PowerPlants(production=prod, name=name)

    n_years = 2050 - 2019
    r_cst = [0.02]
    r_var = n_years * r_cst

    disc_cst = pp.discount(r=r_cst, n=n_years)
    disc_var = pp.discount(r_var)

    with pytest.raises(
        ValueError, match="Discounting cash flows in negative number of year"
    ):
        pp.discount(r=r_cst, n=0.1)

    with pytest.raises(ValueError, match="Discounting cash flows has a wrong format"):
        pp.discount(r=[0.01, 0.02], n=1.1)

    assert np.isclose(disc_cst, disc_var), "Discount is not calculated properly"

    damage = drv_intervals

    loss_cst = pp.financial_losses(
        damages=damage, energy_price=elec_price, r=r_cst, n=n_years
    )

    loss_var = pp.financial_losses(damages=damage, energy_price=elec_price, r=r_var)

    assert np.isclose(
        loss_cst.mean(), loss_var.mean()
    ), "Losses are not calculated properly"

    # Now the same pp in two different ways

    pp2 = PowerPlants(capacity=900913242.0091324, av_rate=1, name=name)
    loss_var2 = pp2.financial_losses(damages=damage, energy_price=elec_price, r=r_var)

    assert np.isclose(
        loss_var.mean(), loss_var2.mean()
    ), "Losses are not calculated properly"

    print("FINISHED DCV TEST SUCCESSFULLY!!!")
