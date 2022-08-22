"""tests for utils.py"""

import pytest

from src.utils import ReLU


class Test_Utils:
    """tests for utils"""

    @pytest.mark.parametrize(
        "input,xoutput",
        [
            (1, 1),
            (1.5, 1.5),
            (0, 0),
            (-1.5, 0),
        ],
    )
    def test_ReLU(self, input, xoutput):
        assert ReLU(input) == xoutput
