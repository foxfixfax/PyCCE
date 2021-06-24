import setuptools
from distutils.core import setup

setup(
    name='pycce',
    version='0.6.0',
    url='',
    license='',
    author='Nikita Onizhuk',
    author_email='onizhuk@uchicago.edu',
    description='A package to compute spin dynamics using CCE method',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy', 'scipy', 'ase', 'pandas'
    ],
)
