import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="MaxSAT tests are disabled on Windows.",
)
