#!/usr/bin/env python3
"""
Test script for deinterlace selector with better error handling.
"""
import asyncio
import sys
from pathlib import Path
from deinterlace import process_video_async


async def safe_process_video(video_path: str, output_path: str):
    """Process video with proper error handling"""
    print(f"Processing: {video_path}")
    print(f"Output: {output_path}")
    print("-" * 60)
    
    try:
        # Check if file exists
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        print("=" * 60)
        recommendation = await process_video_async(video_path, output_path)
        print("=" * 60)
        
        print(f"\n✅ Processing completed successfully!")
        return recommendation
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"Please check that the file exists: {video_path}")
        raise
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the deinterlace module is built correctly.")
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise


async def main():
    if len(sys.argv) < 2:
        print("Usage: python test_video.py <video_file> [output_file]")
        print("Example: python test_video.py myvideo.mp4 output.mov")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output.mov"
    
    try:
        await safe_process_video(video_path, output_path)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
