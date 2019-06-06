# -*- coding: utf-8 -*-
from setuptools import setup
from setuptools import find_packages

setup(
    name='tensorflow-cif-parser',
    version='0.1',
    description='A Crystal Information File (CIF) parser in the TensorFlow'
                ' Datasets API.',
    url='http://github.com/LeanAndMean/tensorflow-cif-parser',
    author='Kevin Ryan',
    author_email='KevinRyan7926+tensorflow-cif-parser@gmail.com',
    license='BSD-3',
    packages=find_packages(),
    install_requires=[
        "tensorflow",
        "ase",
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pyinstrument"],
    include_package_data=True,
    zip_safe=False
)
