#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  setup.py.py

"""

__author__ = 'Welkin'
__date__ = '2020/1/15 02:23'

from setuptools import setup, find_packages

setup(name = 'cvmodels',
      version = '0.1',
      description = 'computer vision models',
      url = 'https://github.com/welkin-feng/ComputerVision',
      author = 'welkin-feng',
      author_email = '',
      license = 'MIT',
      packages = find_packages(),
      platforms = 'any',
      zip_safe = False)
