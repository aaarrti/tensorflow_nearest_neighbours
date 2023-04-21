from setuptools import setup
from setuptools.command.install import install
from setuptools.dist import Distribution


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
    distclass=BinaryDistribution,
    cmdclass={"install": InstallPlatlib},
)
