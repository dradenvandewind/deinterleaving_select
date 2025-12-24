# Create project structure
mkdir -p deinterlace_project/deinterlace
cd deinterlace_project

# Place all the files in their respective directories

# Install dependencies
pip install numpy cython

# Build and install the package
python setup.py build_ext --inplace
# OR
pip install -e .

# Test with a video file
python test_video.py your_video.mp4 output.mov

# For multiple videos
python -c "
import asyncio
from deinterlace.utils import process_multiple_videos

async def main():
    videos = ['video1.mp4', 'video2.mp4', 'video3.mp4']
    results = await process_multiple_videos(videos, './outputs')
    for result in results:
        print(f'Recommended: {result.filter.value}')

asyncio.run(main())
"
