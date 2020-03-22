from setuptools import find_packages, setup
from setuptools_rust import Binding, RustExtension


setup_requires = ['setuptools-rust>=0.10.2']
install_requires = ['numpy']
test_requires = install_requires + ['pytest']

setup(
    name='curved',
    version='0.1.0',
    description='a high performance module for curve simplification',
    rust_extensions=[RustExtension('curved._rustlib', './Cargo.toml', binding=Binding.PyO3)],
    install_requires=install_requires,
    setup_requires=setup_requires,
    test_requires=test_requires,
    packages=find_packages(),
    zip_safe=False,
)
