import numpy as np
from economic_model import WECEconomicModel

def test_cable_costs_two_wecs():
    model = WECEconomicModel()
    coords = np.array([0.0, 0.0, 100.0, 0.0])  # two WECs 100m apart
    cost = model.calculate_cable_costs(coords, n_wecs=2)
    expected_length = 100 * 0.8 + 50  # MST length + shore connection
    expected_cost = expected_length * model.cable_cost_per_meter
    assert np.isclose(cost, expected_cost)
