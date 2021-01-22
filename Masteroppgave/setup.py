#from distutils.core import setup
#from Cython.Build import cythonize

#setup(ext_modules = cythonize('example.pyx'))

######
from setuptools import Extension, setup
from Cython.Build import cythonize

import numpy

from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import os
import numpy

#if os.name == 'nt':
#    opnmp = '/openmp'
#else:
#    opnmp = '-fopenmp'

extensions = [
    Extension(
        name="*",
        sources=["*.pyx"],
        include_dirs=[numpy.get_include()],
        language='c++',
        #extra_compile_args=[opnmp],
        #extra_link_args=[opnmp]
    ),
]

setup(
    ext_modules=cythonize(extensions, annotate=True, force=False, include_path=[numpy.get_include()]),
    include_dirs=[numpy.get_include()],
    cmdclass={'build_ext': build_ext},
)