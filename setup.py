#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re

try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.coding import setup
    setup

# Use UTF-8 if Python 3.
major, minor1, minor2, release, serial = sys.version_info
def read(filename):
    kwargs = {'encoding': 'utf-8'} if major >= 3 else {}
    with open(filename, **kwargs) as f:
        return f.read()

name = 'P3MLens'

# Get current version.
pattern = re.compile('__version__\s*=\s*(\'|")(.*?)(\'|")')
initPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'P3MLens/__init__.py')
version = pattern.findall(read(initPath))[0][1]

setup(
    name=name,
    version=version,
    author='Kun Xu',
    author_email='kunxu.sjtu15@foxmail.com',
    packages=['P3MLens'],
    include_package_data = True,
    url='https://github.com/kunxusjtu/P3MLens',
    license='MIT',
    description='An Accurate P3M Algorithm for Gravitational Lensing Studies in Simulations.',
    long_description=read('README.md'),
    install_requires=['numpy', 'scipy', 'numba', 'astropy],
    tests_require=['pytest', 'pytest-xdist'],
)
