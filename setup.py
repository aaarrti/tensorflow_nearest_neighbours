from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution
from setuptools.command.install import install
from pathlib import Path

__version__ = "0.0.5"

project_name = "tensorflow_nearest_neighbours"


class InstallPlatlib(install):
    def finalize_options(self):
        install.finalize_options(self)
        self.install_lib = self.install_platlib


class BinaryDistribution(Distribution):
    """This class is needed in order to create OS specific wheels."""

    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False


setup(
    name=project_name,
    version=__version__,
    description="Nearest neighbours word embedding computation.",
    author="Artem Sereda",
    author_email="artem.sereda@campus.tu-berlin.de",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    # Contained modules and scripts.
    packages=find_packages(),
    install_requires=[
        "tensorflow; sys_platform != 'darwin'",
        "tensorflow_macos; sys_platform == 'darwin'",
    ],
    # Add in any packaged data.
    include_package_data=True,
    zip_safe=False,
    distclass=BinaryDistribution,
    cmdclass={"install": InstallPlatlib},
    # PyPI package information.
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=["tensorflow", "custom op"],
    python_requires=">=3.8",
)
