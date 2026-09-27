"""
Unit tests for feature engineering functions.
"""

import pytest
import numpy as np
from src.features import haversine_distance, is_near_holiday, get_state_distance


def test_haversine_distance_zero():
    # Distance between identical points is 0
    dist = haversine_distance(-23.5505, -46.6333, -23.5505, -46.6333)
    assert np.isclose(dist, 0.0, atol=1e-3)


def test_haversine_distance_sp_to_rj():
    # Distance between Sao Paulo (-23.5505, -46.6333) and Rio de Janeiro (-22.9068, -43.1729) is ~360 km
    dist = haversine_distance(-23.5505, -46.6333, -22.9068, -43.1729)
    assert 340.0 <= dist <= 380.0


def test_is_near_holiday():
    # Christmas
    assert is_near_holiday("2017-12-24") == 1
    assert is_near_holiday("2017-12-25") == 1
    assert is_near_holiday("2017-12-27") == 1
    # Random date in June
    assert is_near_holiday("2017-06-01") == 0


def test_get_state_distance():
    dist_sp_rj = get_state_distance("SP", "RJ")
    assert 300.0 <= dist_sp_rj <= 500.0

    dist_sp_am = get_state_distance("SP", "AM")
    assert dist_sp_am > 2000.0

    dist_same = get_state_distance("SP", "SP")
    assert np.isclose(dist_same, 0.0, atol=1e-3)
