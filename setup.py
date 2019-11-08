#!/usr/bin/env python3
from pathlib import Path
from setuptools import setup

setup(
    name="siutil",
    version="0.0.1",
    description="Some utilities for sisl that don't quite fit in but are still useful",
    long_description=(Path(__file__).parent / "README.md").open().read(),
    long_description_content_type="text/markdown",
    url="",
    license="LGPLv3",
    packages=["siutil"],
    entry_points={
        'console_scripts':
            ['sgui = siutil.sgui:main']
    },
    install_requires=["sisl", "ase"],
    zip_safe=False,
)
