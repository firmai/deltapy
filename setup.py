from setuptools import setup, Command
import os
import sys


setup(name='deltapy',
      version='0.1.1',
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
            "tensorflow",
            "scikit-learn",
            "scipy",
            "sklearn",
            "statsmodels",
            "numpy",
            "seasonal"],
      zip_safe=False)
