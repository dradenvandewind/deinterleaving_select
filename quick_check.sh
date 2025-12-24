cd /media/erwan/T7/ADN/dentrelace/cython_version/deinterlace

# Test all methods exist
python -c "
from deinterlace.async_processor import DeinterlaceSelectorAsync

selector = DeinterlaceSelectorAsync('vertrezmotion.ts', sample_duration=5)

# Check all required methods
methods = [
    '_analyze_motion',
    '_analyze_motion_simple', 
    '_detect_interlacing',
    '_analyze_complexity',
    '_detect_film_content',
    '_analyze_temporal_consistency',
    '_get_default_motion_data',
    '_generate_reason',
    'analyze_video',
    'recommend_filter',
    'generate_ffmpeg_command',
    'close'
]

print('🔍 Checking all required methods:')
for method in methods:
    has_method = hasattr(selector, method)
    is_callable = callable(getattr(selector, method, None))
    status = '✅' if has_method and is_callable else '❌'
    print(f'{status} {method}: exists={has_method}, callable={is_callable}')

print(f'\\n✅ Selector has close method: {hasattr(selector, \"close\")}')
"

# Run a quick test
python -c "
import asyncio
from deinterlace.async_processor import DeinterlaceSelectorAsync

async def test():
    selector = DeinterlaceSelectorAsync('vertrezmotion.ts', sample_duration=5)
    try:
        print('🔍 Starting analysis...')
        analysis = await selector.analyze_video()
        print(f'✅ Analysis: {analysis}')
        
        rec = await selector.recommend_filter()
        print(f'✅ Recommendation: {rec.filter.value}')
        print(f'  Reason: {rec.reason}')
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        await selector.close()
        print('✅ Cleanup complete')

asyncio.run(test())
"
