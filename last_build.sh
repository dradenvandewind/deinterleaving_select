python -c "
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(
        Extension(
            'deinterlace.core',
            sources=['deinterlace/core.pyx'],
            include_dirs=[np.get_include()],
        ),
        language_level='3'
    )
)
" build_ext --inplace
