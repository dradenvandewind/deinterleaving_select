#!/usr/bin/env python3
"""
Final build script that handles the directory issue correctly.
"""
import os
import sys
import shutil
import subprocess

def main():
    print("🔨 Building deinterlace module...")
    
    # Determine project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if we're inside deinterlace directory
    if os.path.basename(script_dir) == 'deinterlace':
        print("⚠️  You're inside 'deinterlace' directory. Building from parent...")
        
        # Move up one level
        parent_dir = os.path.dirname(script_dir)
        if parent_dir == script_dir:
            print("❌ Cannot move up from root directory")
            return 1
            
        os.chdir(parent_dir)
        print(f"📁 Changed to: {os.getcwd()}")
    
    # Verify core.pyx exists
    core_pyx = "deinterlace/core.pyx"
    if not os.path.exists(core_pyx):
        print(f"❌ {core_pyx} not found!")
        print(f"Current directory: {os.getcwd()}")
        print("Files found:")
        for f in os.listdir('.'):
            print(f"  {f}")
        return 1
    
    print(f"✅ Found {core_pyx}")
    
    # Clean previous builds
    for dir_name in ['build', 'deinterlace.egg-info']:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"🧹 Cleaned {dir_name}")
    
    # Remove any existing .so files
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.so') or file.endswith('.pyd'):
                os.remove(os.path.join(root, file))
                print(f"🧹 Removed old {file}")
    
    # Build with correct paths
    print("\n🚀 Building extension...")
    
    setup_code = '''
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

print(f"Building from: {os.getcwd()}")

setup(
    name="deinterlace",
    packages=["deinterlace"],
    ext_modules=cythonize(
        Extension(
            "deinterlace.core",
            sources=["deinterlace/core.pyx"],
            include_dirs=[np.get_include()],
        ),
        language_level="3",
        build_dir="build"
    )
)
'''
    
    with open("_setup_temp.py", "w") as f:
        f.write('import os\n' + setup_code)
    
    try:
        result = subprocess.run(
            [sys.executable, "_setup_temp.py", "build_ext", "--inplace"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print("❌ Build failed!")
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if 'error' in line.lower() or 'copying' in line:
                        print(f"  {line}")
            if result.stderr:
                for line in result.stderr.split('\n'):
                    if line.strip():
                        print(f"  STDERR: {line}")
        else:
            print("✅ Build completed successfully!")
            
            # Find and verify the .so file
            so_files = []
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if file.endswith('.so'):
                        so_files.append(os.path.join(root, file))
            
            if so_files:
                print("\n📁 Generated files:")
                for so_file in so_files:
                    rel_path = os.path.relpath(so_file)
                    print(f"  ✅ {rel_path}")
                    
                    # Test import
                    print(f"  Testing import...")
                    test_code = f'''
import sys
import os
sys.path.insert(0, '{os.path.dirname(so_file)}')
try:
    import deinterlace.core
    print("    ✅ Import successful!")
    if hasattr(deinterlace.core, 'hello'):
        print(f"    hello() = {{deinterlace.core.hello()}}")
except Exception as e:
    print(f"    ❌ Import failed: {{e}}")
'''
                    
                    import_result = subprocess.run(
                        [sys.executable, "-c", test_code],
                        capture_output=True,
                        text=True
                    )
                    print(import_result.stdout.strip())
            else:
                print("❌ No .so files were created")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    finally:
        # Cleanup
        if os.path.exists("_setup_temp.py"):
            os.remove("_setup_temp.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
