"""``python -m codelor_color`` entry point (same as the ``codelor-color`` command)."""

import sys

from .cli import main

sys.exit(main())
