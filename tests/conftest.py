"""Dummy conftest.py for osc_physrisk_financial.

If you don't know what this is for, just leave it empty.
Read more about conftest.py under:
- https://docs.pytest.org/en/stable/fixture.html
- https://docs.pytest.org/en/stable/writing_plugins.html
"""

import os
import sys

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), "../src")))
