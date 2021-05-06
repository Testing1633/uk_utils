#from distutils.core import setup
from setuptools import setup

setup(
    name='uk_utils',
    packages=['uk_utils'],
    version='0.1.1',
    description='A series of utility functions for my courses',
    author='Ulrich Klauck',
    author_email='ulrich.klauck@hs-aalen.de',
    #url='https://github.com/uk/imutils',
    #download_url='https://github.com/uk/imutils/tarball/0.1',
    keywords=['machine learning', 'image processing'],
    classifiers=[
        'Programming Language :: Python :: 3'
    ],
    scripts=[],
    options={"bdist_wheel": {"universal": True}}
)