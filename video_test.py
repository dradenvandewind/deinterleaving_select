#!/usr/bin/env python3
import asyncio
from deinterlace.async_processor import DeinterlaceSelectorAsync

async def test():
    selector = DeinterlaceSelectorAsync('test.mp4', sample_duration=5)
    try:
        analysis = await selector.analyze_video()
        print(f'✅ Analysis: {analysis}')
        
        rec = await selector.recommend_filter()
        print(f'✅ Recommendation: {rec.filter.value}')
        print(f'  Reason: {rec.reason}')
        
        cmd = selector.generate_ffmpeg_command('output.mov')
        print(f'✅ Command: {cmd[:100]}...')
    finally:
        await selector.close()

asyncio.run(test())

