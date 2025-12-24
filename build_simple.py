#!/usr/bin/env python3
"""
Simple, foolproof build script for deinterlace Cython module.
"""
import os
import sys
import subprocess
import shutil

def main():
    print("🔨 Building deinterlace Cython module...")
    print(f"Python: {sys.version}")
    print(f"CWD: {os.getcwd()}")
    
    # Check for core.pyx
    core_paths = [
        "deinterlace/core.pyx",
        "./deinterlace/core.pyx",
        "core.pyx"
    ]
    
    core_pyx = None
    for path in core_paths:
        if os.path.exists(path):
            core_pyx = os.path.abspath(path)
            print(f"✅ Found: {core_pyx}")
            break
    
    if not core_pyx:
        print("❌ ERROR: core.pyx not found!")
        print("Looked in:", core_paths)
        print("\nFiles in current directory:")
        for f in os.listdir("."):
            print(f"  {f}")
        if os.path.exists("deinterlace"):
            print("\nFiles in deinterlace/ directory:")
            for f in os.listdir("deinterlace"):
                print(f"  {f}")
        return 1
    
    # Create a proper setup.py
    setup_content = '''import sys
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "deinterlace.core",
        sources=["''' + core_pyx + '''"],
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
    
    # Write setup.py in the same directory as core.pyx
    setup_dir = os.path.dirname(core_pyx)
    setup_file = os.path.join(setup_dir, "setup_simple.py")
    
    print(f"📝 Writing setup to: {setup_file}")
    with open(setup_file, "w") as f:
        f.write(setup_content)
    
    # Build
    print("\n🚀 Compiling Cython extension...")
    os.chdir(setup_dir)
    
    try:
        result = subprocess.run(
            [sys.executable, "setup_simple.py", "build_ext", "--inplace"],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            print("❌ Build failed!")
            if result.stdout:
                print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            if result.stderr:
                print("STDERR:", result.stderr[-500:])
            
            # Check for common issues
            print("\n🔍 Checking for issues...")
            
            # Check Cython installation
            try:
                import Cython
                print(f"✅ Cython version: {Cython.__version__}")
            except ImportError:
                print("❌ Cython not installed! Run: pip install cython")
            
            # Check numpy installation
            try:
                import numpy
                print(f"✅ NumPy version: {numpy.__version__}")
            except ImportError:
                print("❌ NumPy not installed! Run: pip install numpy")
            
            return 1
        
        print("✅ Build completed!")
        
        # Show what was created
        print("\n📁 Generated files:")
        build_exts = [".so", ".pyd", ".c"]
        for root, dirs, files in os.walk("."):
            for file in files:
                for ext in build_exts:
                    if file.endswith(ext):
                        full_path = os.path.join(root, file)
                        size = os.path.getsize(full_path)
                        print(f"  {full_path} ({size:,} bytes)")
        
        # Test import
        print("\n🧪 Testing import...")
        try:
            # Add current directory to path
            sys.path.insert(0, ".")
            
            # Clear module cache
            for mod in list(sys.modules.keys()):
                if mod.startswith("deinterlace"):
                    del sys.modules[mod]
            
            import deinterlace.core
            print(f"✅ SUCCESS! Imported deinterlace.core")
            
            # Try to use it
            if hasattr(deinterlace.core, 'hello'):
                print(f"  hello() returns: {deinterlace.core.hello()}")
            
            return 0
            
        except ImportError as e:
            print(f"❌ Import failed: {e}")
            print("\n💡 Try these fixes:")
            print("1. Make sure you're in the right directory")
            print("2. Check if build created a .so file")
            print("3. Try: python -c \"import sys; print(sys.path)\"")
            return 1
            
    except subprocess.TimeoutExpired:
        print("❌ Build timed out after 2 minutes")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    finally:
        # Cleanup
        if os.path.exists(setup_file):
            os.remove(setup_file)
        if os.path.exists("setup_simple.py"):
            os.remove("setup_simple.py")

if __name__ == "__main__":
    sys.exit(main())
