# Activate your deinterlace_env environment
conda activate deinterlace_env

# Verify Python version
python --version
# Should show: Python 3.12.x

# Go to your project directory
cd /media/erwan/T7/ADN/dentrelace/cython_version/deinterlace_debug

# Now rebuild
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

# Check what was created
echo "📁 Files created:"
find . -name "*.so" -o -name "*.c" | sort

# Test import
python -c "
import sys
sys.path.insert(0, '.')
try:
    import deinterlace.core
    print('✅ deinterlace.core imports successfully!')
    print(f'Module location: {deinterlace.core.__file__}')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    import traceback
    traceback.print_exc()
"
