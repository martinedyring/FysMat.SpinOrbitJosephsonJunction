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
        name="utilities",
        sources=["utilities.pyx"],
        include_dirs=[numpy.get_include()],
        language='c++',
        #extra_compile_args=['-fopenmp'],
        #extra_link_args=['-fopenmp'],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-Xpreprocessor", "-fopenmp"],
        extra_link_args=["-Xpreprocessor", "-fopenmp"],
    ),
    Extension(
        name="system_class",
        sources=["system_class.pyx"],
        include_dirs=[numpy.get_include()],
        language='c++',
        # extra_compile_args=['-fopenmp'],
        # extra_link_args=['-fopenmp'],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-Xpreprocessor", "-fopenmp"],
        extra_link_args=["-Xpreprocessor", "-fopenmp"],
    ),
    Extension(
        name="solve_hamiltonian",
        sources=["solve_hamiltonian.pyx"],
        include_dirs=[numpy.get_include()],
        language='c++',
        # extra_compile_args=['-fopenmp'],
        # extra_link_args=['-fopenmp'],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-Xpreprocessor", "-fopenmp"],
        extra_link_args=["-Xpreprocessor", "-fopenmp"],
    ),
    Extension(
        name="__init__",
        sources=["__init__.pyx"],
        include_dirs=[numpy.get_include()],
        language='c++',
        # extra_compile_args=['-fopenmp'],
        # extra_link_args=['-fopenmp'],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-Xpreprocessor", "-fopenmp"],
        extra_link_args=["-Xpreprocessor", "-fopenmp"],
    ),
    Extension(
        name="current_phase_calculations",
        sources=["current_phase_calculations.pyx"],
        include_dirs=[numpy.get_include()],
        language='c++',
        # extra_compile_args=['-fopenmp'],
        # extra_link_args=['-fopenmp'],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-Xpreprocessor", "-fopenmp"],
        extra_link_args=["-Xpreprocessor", "-fopenmp"],
    ),
]

setup(
    include_dirs=[numpy.get_include()],
    ext_modules=cythonize(extensions, annotate=True, force=False),
    cmdclass={'build_ext': build_ext},
)