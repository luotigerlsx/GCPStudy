from setuptools import setup

NAME = 'trainer'
VERSION = '1.0'
REQUIRED_PACKAGES = ['tensorflow-transform==0.6.0']

setup(
    name=NAME,
    version=VERSION,
    install_requires=REQUIRED_PACKAGES,
)
