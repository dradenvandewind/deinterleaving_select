"""
Async wrapper around Cython-optimized deinterlace selector.
Complete version with all required methods.
"""
import asyncio
import subprocess
import re
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# Import from the compiled Cython module
from .core import (
    VideoAnalysis,          # This exists
    DeinterlaceFilter,      # This exists  
    FilterScorer,           # This exists
    AnalysisProcessor,      # This exists
    recommend_filter,       # This exists
    create_video_analysis   # This exists
)

@dataclass
class FilterRecommendation:
    """Filter recommendation with justification"""
    filter: DeinterlaceFilter
    mode: int
    confidence: float
    reason: str
    alternative: Optional[DeinterlaceFilter] = None


class AsyncFFmpegProcessor:
    """Async FFmpeg command processor"""
    
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ffmpeg_")
    
    async def run_command(self, cmd: List[str], timeout: int = 60) -> Tuple[str, str]:
        """Run FFmpeg command asynchronously"""
        loop = asyncio.get_event_loop()
        
        def _run():
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout
            )
        
        try:
            result = await loop.run_in_executor(self.executor, _run)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg command failed: {result.stderr[:200]}")
            return result.stdout, result.stderr
        except asyncio.TimeoutError:
            raise TimeoutError(f"Command timed out after {timeout} seconds")
        except Exception as e:
            raise RuntimeError(f"Command failed: {e}")
    
    def close(self):
        """Cleanup executor"""
        self.executor.shutdown(wait=False)


class DeinterlaceSelectorAsync:
    """
    Intelligent deinterlacing filter selector with Cython optimization
    """
    
    def __init__(self, video_path: str, sample_duration: int = 30):
        """
        Args:
            video_path: Path to video to analyze
            sample_duration: Sample duration for analysis (seconds)
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.sample_duration = min(sample_duration, 60)
        self.analysis: Optional[VideoAnalysis] = None
        self.ffmpeg = AsyncFFmpegProcessor()
        self.scorer = FilterScorer()  # Create instance of FilterScorer
        self.processor = AnalysisProcessor()  # Create instance of AnalysisProcessor
    
    async def analyze_video(self) -> VideoAnalysis:
        """
        Analyze video characteristics asynchronously
        """
        print(f"🔍 Analyzing video: {self.video_path.name}")
        
        # Run analyses concurrently
        tasks = [
            self._analyze_motion(),
            self._detect_interlacing(),
            self._analyze_complexity(),
            self._detect_film_content(),
            self._analyze_temporal_consistency()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results
        motion_data = self._safe_get_result(results[0], self._get_default_motion_data)
        interlace_data = self._safe_get_result(results[1], lambda: {'interlaced_ratio': 0.5})
        complexity = self._safe_get_result(results[2], lambda: 0.15)
        film_detection = self._safe_get_result(results[3], lambda: False)
        temporal = self._safe_get_result(results[4], lambda: 0.5)
        
        # Create VideoAnalysis object using the factory function
        self.analysis = create_video_analysis(
            avg_motion=motion_data['avg'],
            max_motion=motion_data['max'],
            motion_variance=motion_data['variance'],
            scene_changes=motion_data['scene_changes'],
            complexity=complexity,
            interlaced_frames=interlace_data['interlaced_ratio'],
            has_film_content=film_detection,
            temporal_consistency=temporal
        )
        
        return self.analysis
    
    def _safe_get_result(self, result, default_func):
        """Safely get result from asyncio.gather"""
        if isinstance(result, Exception):
            print(f"⚠️ Analysis step failed: {result}")
            return default_func()
        return result
    
    async def _analyze_motion(self) -> Dict[str, float]:
        """Analyze motion using mestimate filter"""
        cmd = [
            'ffmpeg',
            '-hwaccel', 'auto',
            '-i', str(self.video_path),
            '-t', str(min(self.sample_duration, 10)),  # Max 10 seconds for speed
            '-vf', 'mestimate=method=esa,metadata=print:file=-',
            '-f', 'null',
            '-'
        ]
        
        try:
            _, stderr = await self.ffmpeg.run_command(cmd, timeout=30)
            
            # Parse motion metadata
            motion_values = []
            scene_changes = 0
            
            for line in stderr.split('\n'):
                if 'lavfi.mestimate.mb_sad' in line:
                    match = re.search(r'lavfi\.mestimate\.mb_sad=(\d+)', line)
                    if match:
                        motion_values.append(int(match.group(1)))
                
                if 'lavfi.scene_score' in line:
                    scene_changes += 1
            
            if not motion_values:
                # Fallback to simple method
                return await self._analyze_motion_simple()
            
            # Use the AnalysisProcessor from Cython
            result = self.processor.process_motion_data(motion_values)
            result['scene_changes'] += scene_changes
            
            return result
            
        except Exception as e:
            print(f"⚠️ Motion analysis failed, using simple method: {e}")
            return await self._analyze_motion_simple()
    
    async def _analyze_motion_simple(self) -> Dict[str, float]:
        """Simple motion analysis as fallback"""
        cmd = [
            'ffmpeg',
            '-hwaccel', 'auto',
            '-i', str(self.video_path),
            '-t', '2',  # Very short
            '-vf', 'select=not(mod(n\\,10)),tblend=all_mode=difference,metadata=print:file=-',
            '-f', 'null',
            '-'
        ]
        
        try:
            _, stderr = await self.ffmpeg.run_command(cmd, timeout=15)
            
            # Parse difference values
            differences = []
            for line in stderr.split('\n'):
                if 'lavfi.tblend' in line:
                    match = re.search(r'(\d+\.?\d*)', line)
                    if match:
                        try:
                            differences.append(float(match.group(1)))
                        except:
                            pass
            
            if differences:
                avg_diff = np.mean(differences)
                max_diff = np.max(differences) if differences else 0
                
                # Convert to motion estimate
                motion_estimate = min(avg_diff / 50.0, 1.0)
                
                return {
                    'avg': motion_estimate * 0.2,
                    'max': min(max_diff / 100.0, 0.5),
                    'variance': 0.05,
                    'scene_changes': max(1, int(len(differences) / 5))
                }
            else:
                return self._get_default_motion_data()
                
        except Exception as e:
            print(f"⚠️ Simple motion analysis failed: {e}")
            return self._get_default_motion_data()
    
    async def _detect_interlacing(self) -> Dict[str, float]:
        """Detect interlacing ratio using idet"""
        cmd = [
            'ffmpeg',
            '-hwaccel', 'auto',
            '-i', str(self.video_path),
            '-t', str(min(self.sample_duration, 5)),  # Max 5 seconds
            '-vf', 'idet',
            '-f', 'null',
            '-'
        ]
        
        try:
            _, stderr = await self.ffmpeg.run_command(cmd, timeout=20)
            # Use the AnalysisProcessor from Cython
            return self.processor.parse_interlace_data(stderr)
            
        except Exception as e:
            print(f"⚠️ Interlacing detection failed: {e}")
            return {'interlaced_ratio': 0.5}
    
    async def _analyze_complexity(self) -> float:
        """Analyze spatial complexity of the image"""
        cmd = [
            'ffmpeg',
            '-hwaccel', 'auto',
            '-i', str(self.video_path),
            '-t', str(min(self.sample_duration, 5)),  # Max 5 seconds
            '-vf', 'select=not(mod(n\\,50)),signalstats',
            '-f', 'null',
            '-'
        ]
        
        try:
            _, stderr = await self.ffmpeg.run_command(cmd, timeout=20)
            # Use the AnalysisProcessor from Cython
            return self.processor.parse_complexity_data(stderr)
            
        except Exception as e:
            print(f"⚠️ Complexity analysis failed: {e}")
            return 0.15
    
    async def _detect_film_content(self) -> bool:
        """Detect if content is film (24fps) telecined to 50i"""
        cmd = [
            'ffmpeg',
            '-hwaccel', 'auto',
            '-i', str(self.video_path),
            '-t', str(min(self.sample_duration, 5)),  # Max 5 seconds
            '-vf', 'pullup,idet',
            '-f', 'null',
            '-'
        ]
        
        try:
            _, stderr = await self.ffmpeg.run_command(cmd, timeout=20)
            
            # Look for telecine patterns
            repeated_frames = 0
            
            for line in stderr.split('\n'):
                if 'repeated' in line.lower():
                    repeated_frames += 1
            
            total_frames = min(self.sample_duration, 5) * 25
            return (repeated_frames / max(total_frames, 1)) > 0.2
            
        except Exception as e:
            print(f"⚠️ Film content detection failed: {e}")
            return False
    
    async def _analyze_temporal_consistency(self) -> float:
        """Analyze temporal consistency between frames"""
        cmd = [
            'ffmpeg',
            '-hwaccel', 'auto',
            '-i', str(self.video_path),
            '-t', str(min(self.sample_duration, 5)),  # Max 5 seconds
            '-vf', 'tblend=all_mode=difference,metadata=print:file=-',
            '-f', 'null',
            '-'
        ]
        
        try:
            _, stderr = await self.ffmpeg.run_command(cmd, timeout=20)
            
            # Measure difference between consecutive frames
            differences = []
            
            for line in stderr.split('\n'):
                if 'lavfi.tblend' in line:
                    match = re.search(r'(\d+\.?\d*)', line)
                    if match:
                        differences.append(float(match.group(1)))
            
            if differences:
                # Low variance = good temporal consistency
                return float(1.0 - min(np.var(differences) / 1000.0, 1.0))
            else:
                return 0.5
                
        except Exception as e:
            print(f"⚠️ Temporal consistency analysis failed: {e}")
            return 0.5
    
    def _get_default_motion_data(self) -> Dict[str, float]:
        """Get default motion data"""
        return {
            'avg': 0.1,
            'max': 0.2,
            'variance': 0.05,
            'scene_changes': 5
        }
    
    def _generate_reason(self, filter: DeinterlaceFilter, analysis: VideoAnalysis) -> str:
        """Generate explanation for recommendation"""
        reasons = []
        
        if filter == DeinterlaceFilter.YADIF:
            reasons.append("Versatile filter suited for content")
            if analysis.avg_motion < 0.15:
                reasons.append("low to medium motion detected")
            if analysis.complexity < 0.3:
                reasons.append("moderate spatial complexity")
        
        elif filter == DeinterlaceFilter.BWDIF:
            reasons.append("Better quality than yadif recommended")
            if analysis.complexity > 0.15:
                reasons.append("high spatial complexity detected")
            reasons.append("optimal detail preservation")
        
        elif filter == DeinterlaceFilter.ESTDIF:
            reasons.append("Optimized for fast motion")
            if analysis.avg_motion > 0.15:
                reasons.append(f"high motion detected ({analysis.avg_motion:.3f})")
            scene_rate = analysis.scene_changes / (self.sample_duration * 25)
            if scene_rate > 0.05:
                reasons.append("multiple scene changes")
        
        elif filter == DeinterlaceFilter.W3FDIF:
            reasons.append("Optimized for film content")
            if analysis.has_film_content:
                reasons.append("telecined content detected")
            if analysis.temporal_consistency > 0.7:
                reasons.append("excellent temporal consistency")
                
        elif filter == DeinterlaceFilter.NNEDI:
            reasons.append("Maximum quality with neural network")
            if analysis.complexity > 0.2:
                reasons.append("fine details and high complexity")
            if analysis.avg_motion < 0.15:
                reasons.append("motion suitable for neural processing")
            reasons.append("exceptional edge preservation")
            
        return ", ".join(reasons)
    
    async def recommend_filter(self) -> FilterRecommendation:
        """
        Recommend the best filter based on analysis
        """
        if self.analysis is None:
            await self.analyze_video()
        
        print(f"\n📊 Analysis results:")
        print(f"  Average motion: {self.analysis.avg_motion:.3f}")
        print(f"  Max motion: {self.analysis.max_motion:.3f}")
        print(f"  Complexity: {self.analysis.complexity:.3f}")
        print(f"  Interlacing: {self.analysis.interlaced_frames:.1%}")
        print(f"  Film content: {self.analysis.has_film_content}")
        
        # Use the recommend_filter function from Cython
        cython_recommendation = recommend_filter(self.analysis)
        
        # Handle both dictionary and object returns
        if isinstance(cython_recommendation, dict):
            # It's a dictionary - convert to FilterRecommendation
            print("⚠️  Cython returned dictionary, converting to object...")
            filter_obj = cython_recommendation.get('filter', DeinterlaceFilter.YADIF)
            mode = cython_recommendation.get('mode', 1)
            confidence = cython_recommendation.get('score', 0.5)  # Note: 'score' not 'confidence'
            alternative = cython_recommendation.get('alternative', None)
            
            # Generate reason
            reason = self._generate_reason(filter_obj, self.analysis)
            
            return FilterRecommendation(
                filter=filter_obj,
                mode=mode,
                confidence=confidence,
                reason=reason,
                alternative=alternative
            )
        else:
            # It's already a FilterRecommendation object
            return cython_recommendation
    
    def generate_ffmpeg_command(self, output_path: str, 
                               codec: str = 'prores_ks',
                               profile: int = 3) -> str:
        """
        Generate FFmpeg command with recommended filter
        """
        if self.analysis is None:
            raise ValueError("Video must be analyzed first")
        
        # Get recommendation synchronously
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        rec = loop.run_until_complete(self.recommend_filter())
        
        # Build filter
        if rec.filter == DeinterlaceFilter.YADIF:
            vf = f"yadif={rec.mode}"
        elif rec.filter == DeinterlaceFilter.BWDIF:
            vf = f"bwdif={rec.mode}"
        elif rec.filter == DeinterlaceFilter.ESTDIF:
            vf = f"estdif=mode={rec.mode}"
        elif rec.filter == DeinterlaceFilter.NNEDI:
            vf = "nnedi=weights=nnedi3_weights.bin"
        else:  # W3FDIF
            vf = "w3fdif"
        
        # Add pixel format for ProRes
        if codec == 'prores_ks':
            vf += ",format=yuv422p10le"
        
        cmd = [
            'ffmpeg',
            '-hwaccel', 'auto',
            '-i', str(self.video_path),
            '-vf', vf,
            '-c:v', codec,
            '-profile:v', str(profile),
            '-c:a', 'copy',
            '-y',
            output_path
        ]
        
        return ' '.join(cmd)
    
    async def close(self):
        """Cleanup resources"""
        if hasattr(self, 'ffmpeg') and self.ffmpeg:
            self.ffmpeg.close()