"""
Tests for utility functions
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from eth_deanon.utils.helpers import (
    format_timestamp,
    calculate_time_delta,
    normalize_address,
    calculate_percentiles,
    filter_outliers,
    create_time_windows,
    calculate_rolling_metrics,
    format_currency,
    calculate_growth_rate
)

class TestHelpers(unittest.TestCase):
    def test_format_timestamp(self):
        timestamp = 1609459200  # 2021-01-01 00:00:00
        expected = "2021-01-01 00:00:00"
        self.assertEqual(format_timestamp(timestamp), expected)

    def test_calculate_time_delta(self):
        timestamps = [1609459200, 1609545600, 1609632000]  # 1 day intervals
        expected = [86400, 86400]  # 24 hours in seconds
        np.testing.assert_array_equal(calculate_time_delta(timestamps), expected)

    def test_normalize_address(self):
        address = "0x1234ABCDEF"
        expected = "0x1234abcdef"
        self.assertEqual(normalize_address(address), expected)

    def test_calculate_percentiles(self):
        data = [1, 2, 3, 4, 5]
        expected = {25: 2.0, 50: 3.0, 75: 4.0}
        self.assertEqual(calculate_percentiles(data), expected)

    def test_filter_outliers(self):
        data = np.array([1, 2, 3, 4, 5, 100])
        filtered = filter_outliers(data)
        self.assertEqual(len(filtered), 5)
        self.assertTrue(all(x <= 5 for x in filtered))

    def test_create_time_windows(self):
        start_date = datetime(2021, 1, 1)
        end_date = datetime(2021, 1, 4)
        windows = create_time_windows(start_date, end_date, 1)
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0][0], start_date)
        self.assertEqual(windows[-1][1], end_date)

    def test_calculate_rolling_metrics(self):
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        result = calculate_rolling_metrics(df, 3)
        self.assertIn('rolling_mean', result.columns)
        self.assertIn('rolling_std', result.columns)

    def test_format_currency(self):
        self.assertEqual(format_currency(1000000000), "$1.00B")
        self.assertEqual(format_currency(1000000), "$1.00M")
        self.assertEqual(format_currency(1000), "$1.00K")
        self.assertEqual(format_currency(100), "$100.00")

    def test_calculate_growth_rate(self):
        self.assertAlmostEqual(
            calculate_growth_rate(100, 200, 1),
            100.0,
            places=2
        )
        self.assertEqual(calculate_growth_rate(0, 100, 1), 0)
        self.assertEqual(calculate_growth_rate(100, 200, 0), 0)

if __name__ == '__main__':
    unittest.main() 