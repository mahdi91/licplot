from Cython.Build import cythonize


def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration("lic", parent_package, top_path)

    return config
