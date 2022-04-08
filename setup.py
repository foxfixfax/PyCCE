from distutils.core import setup

import setuptools

setup(
    name='pycce',
    version='1.0.0',
    url='',
    license='',
    author='Nikita Onizhuk',
    author_email='onizhuk@uchicago.edu',
    description='A package to compute spin dynamics using CCE method',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy', 'scipy', 'ase', 'pandas'
    ],
    include_package_data=True,
    package_data={'': ['bath/*.txt']},
)
