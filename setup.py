"""
Setup script for Cython-optimized deinterlace selector.
Compatible with Python 3.7+
"""
import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy as np

# Check Python version
if sys.version_info < (3, 7):
    raise RuntimeError("Python 3.7 or higher required")

# Get the absolute path to the directory containing setup.py
setup_dir = os.path.dirname(os.path.abspath(__file__))

# Define Cython extension - adjust the path to core.pyx
core_pyx_path = os.path.join(setup_dir, "deinterlace", "core.pyx")

if not os.path.exists(core_pyx_path):
    # Try alternative location
    core_pyx_path = os.path.join(setup_dir, "core.pyx")
    if not os.path.exists(core_pyx_path):
        print(f"ERROR: Could not find core.pyx at {core_pyx_path}")
        print(f"Current directory: {setup_dir}")
        print("Files in current directory:")
        for f in os.listdir(setup_dir):
            print(f"  - {f}")
        sys.exit(1)

extensions = [
    Extension(
        "deinterlace.core",
        sources=[core_pyx_path],  # Use absolute path
        include_dirs=[np.get_include()],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        extra_compile_args=['-O3', '-march=native', '-ffast-math'] if sys.platform != 'win32' else ['/O2'],
        language="c",
    )
]

# Custom build class to handle Cython
class BuildExt(build_ext):
    def run(self):
        # Check if Cython is available
        try:
            from Cython.Build import cythonize
            # Cythonize the extensions
            self.extensions = cythonize(
                self.extensions,
                compiler_directives={
                    'language_level': 3,
                    'boundscheck': False,
                    'wraparound': False,
                    'initializedcheck': False,
                    'cdivision': True,
                    'nonecheck': False,
                },
                annotate=False
            )
        except ImportError:
            print("Cython not found, trying to build from .c file")
            # Check if .c file exists
            c_file = core_pyx_path.replace('.pyx', '.c')
            if os.path.exists(c_file):
                for ext in self.extensions:
                    ext.sources = [c_file]
            else:
                print(f"ERROR: Neither Cython nor {c_file} found")
                print("Please install Cython: pip install cython")
                sys.exit(1)
        
        super().run()

# Read README for long description
long_description = ""
readme_path = os.path.join(setup_dir, "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()

setup(
    name="deinterlace-selector",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Async deinterlacing filter selector with Cython optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["deinterlace"],
    ext_modules=extensions,
    cmdclass={'build_ext': BuildExt},
    install_requires=[
        'numpy>=1.21.0',
        'Cython>=3.0.0',
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Video",
    ],
    keywords="deinterlace, video, ffmpeg, cython, async",
)