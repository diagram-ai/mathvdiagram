#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    __init__.py
    ~~~~~~~~~~~

    Pipeline for Math Diagram Benchmark Generation

    :copyright: (c) 2026 by Harish Kashyap, Sriram CR, Sanyukta Tuti.
    :license: see LICENSE for more details.
"""

__title__ = 'mathvdiagram'
__version__ = '0.0.1'
__author__ = 'Harish Kashyap, Sriram CR, Sanyukta Tuti'

import logging
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s", force=True)

# Re-export dataset_helper for convenience:
#   from mathvdiagram import dataset_helper
#   from mathvdiagram.dataset_helper import run_full_classification
from . import dataset_helper  # noqa: F401
