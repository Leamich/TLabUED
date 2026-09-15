"""Tests for the final-number aggregation the README quotes.

Pure pandas: no JAX needed, so these run everywhere.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tlab_ued.analysis import ema, final_table, per_level_table


def _runs() -> pd.DataFrame:
    """Two seeds of one method, three evaluations each, rows deliberately shuffled."""
    rows = []
    for seed, values in ((0, [0.0, 1.0, 1.0]), (1, [1.0, 0.0, 0.5])):
        for i, value in enumerate(values):
            rows.append(
                {
                    "run_name": "method",
                    "seed": seed,
                    "num_updates": 250 * (i + 1),
                    "solve_rate/mean": value,
                    "solve_rate/SixteenRooms": value,
                }
            )
    return pd.DataFrame(rows).sample(frac=1.0, random_state=0)


def test_ema_is_seeded_with_the_first_value():
    # 0 -> 0.8*0 + 0.2*1 = 0.2 -> 0.8*0.2 + 0.2*1 = 0.36
    assert ema([0.0, 1.0, 1.0], 0.8) == pytest.approx(0.36)


def test_final_table_ema_follows_update_order_not_row_order():
    table = final_table(_runs(), ema_gamma=0.8)
    # seeds: 0.36 and 1 -> 0.8 -> 0.8*0.8 + 0.2*0.5 = 0.74
    assert table.loc[0, "mean"] == pytest.approx((0.36 + 0.74) / 2)
    assert table.loc[0, "count"] == 2


def test_last_k_is_unchanged_without_ema():
    table = final_table(_runs(), last_k=2)
    assert table.loc[0, "mean"] == pytest.approx((1.0 + 0.25) / 2)


def test_per_level_table_uses_the_same_aggregation():
    table = per_level_table(_runs(), levels=("SixteenRooms",), ema_gamma=0.8)
    assert table.loc["method", "SixteenRooms"] == pytest.approx((0.36 + 0.74) / 2)
