#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import re

try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup

# Handle encoding
major, minor1, minor2, release, serial = sys.version_info
if major >= 3:
    def rd(filename):
        f = open(filename, encoding="utf-8")
        r = f.read()
        f.close()
        return r
else:
    def rd(filename):
        f = open(filename)
        r = f.read()
        f.close()
        return r

vre = re.compile("__version__ = \"(.*?)\"")
m = rd(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "starsam", "__init__.py"))
version = vre.findall(m)[0]

setup(
    name="starsam",
    version=version,
    author="Alexander J. Dittmann",
    author_email="dittmann@ias.edu",
    packages=["starsam"],
    description="Semi-analytical models of stellar evolution in AGN disks and other dense environments.",
    install_requires=["numpy","numba","scipy"],
)
