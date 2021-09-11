from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    rust_extensions=[RustExtension('curved._rustlib', binding=Binding.PyO3)]
)
