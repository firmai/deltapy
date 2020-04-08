from setuptools import setup, Command
import os
import sys


setup(name='deltapy',
      version='0.0.2',
      description='Data Transformation and Augmentation',
      url='https://github.com/firmai/deltapy',
      author='snowde',
      author_email='d.snow@firmai.org',
      license='MIT',
      packages=['deltapy'],
      install_requires=[
            "fbprophet",
            "pandas",
            "pykalman",
            "tsaug",
            "gplearn",
            "ta",
            "scikit-learn",
            "scipy",
            "sklearn",
            "statsmodels",
            "numpy"],
      zip_safe=False)
