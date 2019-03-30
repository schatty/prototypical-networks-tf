import setuptools
from setuptools import setup


setup(name='prototf',
      version='0.0.1',
      license='MIT',
      packages=setuptools.find_packages(),
      install_requires=[
            'tensorflow==2.0.0-alpha0',
            'Pillow'
      ])