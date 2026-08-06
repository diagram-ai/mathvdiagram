#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    __init__.py
    ~~~~~~~~~~~

    Pipeline for Math Diagram Benchmark Generation

    :copyright: (c) 2026 Diagram AI.
    :license: Apache-2.0, see LICENSE for more details.
"""

__title__ = 'mathvdiagram'
__version__ = '0.1.0'
__author__ = 'Harish Kashyap, Sriram CR, Sanyukta Tuti, Aryan Mistry, Kiran'

# Re-export dataset_helper for convenience:
#   from mathvdiagram import dataset_helper
#   from mathvdiagram.dataset_helper import run_full_classification
from . import dataset_helper  # noqa: F401
