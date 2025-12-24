"""
Utility functions for deinterlace processing.
"""
import asyncio
from typing import List
from .async_processor import DeinterlaceSelectorAsync, FilterRecommendation

async def process_video_async(video_path: str, output_path: str) -> FilterRecommendation:
    """
    Process a single video asynchronously with proper cleanup
    """
    selector = None
    try:
        selector = DeinterlaceSelectorAsync(video_path, sample_duration=30)
        
        print("=" * 60)
        await selector.analyze_video()
        print("=" * 60)
        
        recommendation = await selector.recommend_filter()
        
        print(f"\n✅ Recommendation:")
        print(f"  Filter: {recommendation.filter.value}")
        print(f"  Mode: {recommendation.mode} ({'50fps' if recommendation.mode == 1 else '25fps'})")
        print(f"  Confidence: {recommendation.confidence:.1%}")
        print(f"  Reason: {recommendation.reason}")
        if recommendation.alternative:
            print(f"  Alternative: {recommendation.alternative.value}")
        
        print(f"\n🎬 FFmpeg command:")
        cmd = selector.generate_ffmpeg_command(output_path)
        print(f"  {cmd}")
        
        return recommendation
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        raise
    finally:
        # Ensure cleanup happens even if there's an error
        if selector:
            await selector.close()

async def process_multiple_videos(video_paths: List[str], output_dir: str):
    """
    Process multiple videos concurrently
    """
    tasks = []
    for i, video_path in enumerate(video_paths):
        output_path = f"{output_dir}/output_{i}.mov"
        tasks.append(process_video_async(video_path, output_path))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
