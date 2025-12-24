conda create -n deinterlace_env python=3.12 cython numpy -y

# Activate it
conda activate deinterlace_env

# Verify Python version
python --version

./RebuildAllModule.py
