"""tests for utils.py"""

import pytest

from src.utils import ReLU, ReLU_deriv


class Test_Helpers:
    """tests for Helpers"""

    @pytest.mark.parametrize(
        "input,xoutput",
        [
            (0.5, 0.5),
            (1, 1),
            (1.5, 1.5),
            (0, 0),
            (-1.5, 0),
        ],
    )
    def test_ReLU(self, input, xoutput):
        assert ReLU(input) == xoutput

    @pytest.mark.parametrize(
        "input,xoutput",
        [
            (0.5, 1),
            (1, 1),
            (1.5, 1),
            (2, 1),
            (0, 0),
            (-1.5, 0),
        ],
    )
    def test_ReLU_deriv(self, input, xoutput):
        assert ReLU_deriv(input) == xoutput
