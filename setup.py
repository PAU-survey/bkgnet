import os
from setuptools import setup

# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "bkgnet",
    version = "0.1.0",
    author = "Laura Cabayol",
    author_email = "lcabayol@ifae.es",
    description = ("Background estimation from PAUS with neural networks."),
    keywords = "astronomy",
    url = "https://gitlab.pic.es/pau/bkgnet",
    license="GPLv3",
    packages=['torch', 'numpy', 'pandas'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Astronomy",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ],
)
