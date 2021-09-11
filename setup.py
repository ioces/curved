from setuptools import find_packages, setup
from setuptools_rust import Binding, RustExtension


install_requires = ['numpy']

setup(
    name='curved',
    version='0.1.1',
    description='a high performance module for curve simplification',
    rust_extensions=[RustExtension('curved._rustlib', binding=Binding.PyO3)],
    install_requires=install_requires,
    packages=find_packages(),
    zip_safe=False,
)
