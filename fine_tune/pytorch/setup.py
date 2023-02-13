"""Setup for pip package."""

import unittest
from setuptools import find_namespace_packages
from setuptools import setup


# def _parse_requirements(requirements_txt_path):
#   with open(requirements_txt_path) as fp:
#     return fp.read().splitlines()
#
#
# def test_suite():
#   test_loader = unittest.TestLoader()
#   all_tests = test_loader.discover('jax_privacy',
#                                    pattern='*_test.py')
#   return all_tests

setup(
    name='fine_tune',
    version='0.0.0',
    description='DP-DL in PyTorch.',
    # url='https://github.com/deepmind/jax_privacy',
    # author='DeepMind',
    # author_email='jax-privacy-dev@deepmind.com',
    # # Contained modules and scripts.
    # packages=find_namespace_packages(exclude=['*_test.py', 'experiments.*']),
    # install_requires=_parse_requirements('requirements.txt'),
    # requires_python='>=3.7',
    # platforms=['any'],
    # license='Apache 2.0',
    # test_suite='setup.test_suite',
    # include_package_data=True,
    # zip_safe=False,
)
