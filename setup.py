# Copyright (C) 2019 Laura Cabayol, Martin B. Eriksen
# This file is part of BKGnet <https://github.com/PAU-survey/bkgnet>.
#
# BKGnet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BKGnet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BKGnet.  If not, see <http://www.gnu.org/licenses/>.
import os
from setuptools import setup

# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# This requires PyTorch, which can not be automatically be installed
# using pip.

setup(
    name = "bkgnet",
    version = "0.1.0",
    author = "Laura Cabayol",
    author_email = "lcabayol@ifae.es",
    description = ("Background estimation from PAUS with neural networks."),
    keywords = "astronomy",
    url = "https://gitlab.pic.es/pau/bkgnet",
    license="GPLv3",
    packages=['bkgnet'],
    install_requires=['numpy', 'pandas'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Astronomy",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ],
)
