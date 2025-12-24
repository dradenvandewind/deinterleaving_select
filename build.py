#!/usr/bin/env python3
"""
Simple build script for deinterlace Cython module.
"""
import os
import sys
import subprocess
import shutil

def build_cython():
    print("🔨 Building Cython extension...")
    
    # Check if core.pyx exists
    core_pyx = "deinterlace/core.pyx"
    if not os.path.exists(core_pyx):
        print(f"❌ ERROR: {core_pyx} not found!")
        print("Current directory:", os.getcwd())
        print("Files in deinterlace/:", os.listdir("deinterlace") if os.path.exists("deinterlace") else "No deinterlace directory")
        return False
    
    # Create minimal setup.py
    setup_content = '''from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "deinterlace.core",
        sources=["deinterlace/core.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        extra_compile_args=['-O3'] if sys.platform != 'win32' else ['/O2'],
        language="c",
    )
]

setup(
    name="deinterlace",
    ext_modules=cythonize(extensions, language_level="3"),
)
'''
    
    with open("setup_build.py", "w") as f:
        f.write(setup_content)
    
    # Build
    try:
        result = subprocess.run(
            [sys.executable, "setup_build.py", "build_ext", "--inplace"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Build successful!")
            
            # Check what was created
            print("\n📁 Generated files:")
            for root, dirs, files in os.walk("."):
                for file in files:
                    if file.endswith((".so", ".pyd", ".c")):
                        print(f"  {os.path.join(root, file)}")
            
            # Test import
            print("\n🧪 Testing import...")
            try:
                # Clear any cached imports
                if 'deinterlace.core' in sys.modules:
                    del sys.modules['deinterlace.core']
                
                import deinterlace.core
                print("✅ deinterlace.core imports successfully!")
                return True
            except ImportError as e:
                print(f"❌ Import failed: {e}")
                return False
        else:
            print("❌ Build failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Build error: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists("setup_build.py"):
            os.remove("setup_build.py")

if __name__ == "__main__":
    success = build_cython()
    sys.exit(0 if success else 1)
