import setuptools
from setuptools import setup

setup(
    name='pycce',
    version='1.1.0',
    url='',
    license='',
    author='Nikita Onizhuk',
    author_email='onizhuk@uchicago.edu',
    description='A package to compute spin dynamics using CCE method',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy', 'scipy', 'ase', 'pandas', 'numba'
    ],
    include_package_data=True,
    package_data={'': ['bath/*.txt']},
)
