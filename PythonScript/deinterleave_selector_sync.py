#!/usr/bin/env python3
"""
Automatic deinterlacing filter selection algorithm
based on motion analysis and video characteristics.
Async version with Cython optimization.
"""

import asyncio
import subprocess
import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import cython

# Cython optimization decorator
@cython.cclass
class DeinterlaceSelectorAsync:
    pass

# Cython type definitions for performance
ctypedef float float32
ctypedef int int32


class DeinterlaceFilter(Enum):
    """Available deinterlacing filters"""
    YADIF = "yadif"
    BWDIF = "bwdif"
    ESTDIF = "estdif"
    W3FDIF = "w3fdif"
    NNEDI = "nnedi"


@dataclass
class VideoAnalysis:
    """Video analysis results"""
    avg_motion: float32
    max_motion: float32
    motion_variance: float32
    scene_changes: int32
    complexity: float32
    interlaced_frames: float32
    has_film_content: cython.bint
    temporal_consistency: float32


@dataclass
class FilterRecommendation:
    """Filter recommendation with justification"""
    filter: DeinterlaceFilter
    mode: int32
    confidence: float32
    reason: str
    alternative: Optional[DeinterlaceFilter] = None


class AsyncFFmpegProcessor:
    """Async FFmpeg command processor"""
    
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def run_command(self, cmd: List[str], timeout: int = 300) -> Tuple[str, str]:
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
            return result.stdout, result.stderr
        except asyncio.TimeoutError:
            raise TimeoutError(f"Command timed out after {timeout} seconds")
        except Exception as e:
            raise RuntimeError(f"Command failed: {e}")
    
    def close(self):
        """Cleanup executor"""
        self.executor.shutdown(wait=True)


class DeinterlaceSelectorAsync:
    """
    Intelligent deinterlacing filter selector
    based on video content analysis
    """
    
    # Decision thresholds
    THRESHOLDS = {
        'high_motion': 0.15,      # High motion
        'low_motion': 0.05,       # Low motion
        'high_complexity': 0.3,   # High spatial complexity
        'scene_change_rate': 0.05, # Scene change rate
        'film_confidence': 0.7,   # Film detection confidence
    }
    
    def __init__(self, video_path: str, sample_duration: int = 30):
        """
        Args:
            video_path: Path to video to analyze
            sample_duration: Sample duration for analysis (seconds)
        """
        self.video_path = Path(video_path)
        self.sample_duration = sample_duration
        self.analysis: Optional[VideoAnalysis] = None
        self.ffmpeg = AsyncFFmpegProcessor()
    
    async def analyze_video(self) -> VideoAnalysis:
        """
        First pass: motion and characteristics analysis
        """
        print(f"🔍 Analyzing video: {self.video_path.name}")
        
        # Run all analyses concurrently
        tasks = [
            self._analyze_motion(),
            self._detect_interlacing(),
            self._analyze_complexity(),
            self._detect_film_content(),
            self._analyze_temporal_consistency()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results
        motion_data = results[0] if not isinstance(results[0], Exception) else self._get_default_motion_data()
        interlace_data = results[1] if not isinstance(results[1], Exception) else {'interlaced_ratio': 0.5}
        complexity = results[2] if not isinstance(results[2], Exception) else 0.15
        film_detection = results[3] if not isinstance(results[3], Exception) else False
        temporal = results[4] if not isinstance(results[4], Exception) else 0.5
        
        self.analysis = VideoAnalysis(
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
    
    async def _analyze_motion(self) -> Dict[str, float32]:
        """Analyze motion using mestimate filter"""
        cmd = [
            'ffmpeg',
            '-i', str(self.video_path),
            '-t', str(self.sample_duration),
            '-vf', 'mestimate=method=esa,metadata=print:file=-',
            '-f', 'null',
            '-'
        ]
        
        try:
            _, stderr = await self.ffmpeg.run_command(cmd)
            
            # Parse motion metadata
            motion_values = []
            scene_changes = 0
            
            for line in stderr.split('\n'):
                # Look for motion values
                if 'lavfi.mestimate.mb_sad' in line:
                    match = re.search(r'lavfi\.mestimate\.mb_sad=(\d+)', line)
                    if match:
                        motion_values.append(int(match.group(1)))
                
                # Detect scene changes
                if 'lavfi.scene' in line or (motion_values and motion_values[-1] > 50000):
                    scene_changes += 1
            
            if not motion_values:
                # Fallback: use select and setpts
                return await self._analyze_motion_fallback()
            
            motion_array = np.array(motion_values, dtype=np.float32)
            
            return {
                'avg': np.mean(motion_array) / 10000.0,  # Normalize
                'max': np.max(motion_array) / 10000.0,
                'variance': np.var(motion_array) / 100000000.0,
                'scene_changes': scene_changes
            }
            
        except Exception:
            return await self._analyze_motion_fallback()
    
    async def _analyze_motion_fallback(self) -> Dict[str, float32]:
        """Alternative motion analysis method"""
        # Use mpdecimate to estimate motion
        cmd = [
            'ffmpeg',
            '-i', str(self.video_path),
            '-t', str(self.sample_duration),
            '-vf', 'mpdecimate,metadata=print:file=-',
            '-f', 'null',
            '-'
        ]
        
        try:
            _, stderr = await self.ffmpeg.run_command(cmd)
            
            # Count duplicate frames (low motion)
            dropped = stderr.count('drop')
            total_frames = self.sample_duration * 25  # Assume 25fps
            
            motion_estimate = 1.0 - (dropped / max(total_frames, 1))
            
            return {
                'avg': motion_estimate * 0.2,
                'max': motion_estimate * 0.4,
                'variance': 0.05,
                'scene_changes': max(1, int(total_frames / 100))
            }
        except Exception:
            # Conservative default values
            return {
                'avg': 0.1,
                'max': 0.2,
                'variance': 0.05,
                'scene_changes': 5
            }
    
    def _get_default_motion_data(self) -> Dict[str, float32]:
        """Get default motion data when analysis fails"""
        return {
            'avg': 0.1,
            'max': 0.2,
            'variance': 0.05,
            'scene_changes': 5
        }
    
    async def _detect_interlacing(self) -> Dict[str, float32]:
        """Detect interlacing ratio using idet"""
        cmd = [
            'ffmpeg',
            '-i', str(self.video_path),
            '-t', str(self.sample_duration),
            '-vf', 'idet',
            '-f', 'null',
            '-'
        ]
        
        try:
            _, stderr = await self.ffmpeg.run_command(cmd)
            
            # Parse idet results
            tff = bff = progressive = 0
            
            for line in stderr.split('\n'):
                if 'Multi frame detection' in line:
                    match_tff = re.search(r'TFF:\s*(\d+)', line)
                    match_bff = re.search(r'BFF:\s*(\d+)', line)
                    match_prog = re.search(r'Progressive:\s*(\d+)', line)
                    
                    if match_tff:
                        tff = int(match_tff.group(1))
                    if match_bff:
                        bff = int(match_bff.group(1))
                    if match_prog:
                        progressive = int(match_prog.group(1))
            
            total = tff + bff + progressive
            if total == 0:
                return {'interlaced_ratio': 0.5}  # Unknown, assume interlaced
            
            interlaced_ratio = (tff + bff) / total
            
            return {
                'interlaced_ratio': interlaced_ratio,
                'tff': tff,
                'bff': bff,
                'progressive': progressive
            }
            
        except Exception:
            return {'interlaced_ratio': 0.5}
    
    async def _analyze_complexity(self) -> float32:
        """Analyze spatial complexity of the image"""
        cmd = [
            'ffmpeg',
            '-i', str(self.video_path),
            '-t', str(self.sample_duration),
            '-vf', 'select=not(mod(n\\,25)),signalstats',
            '-f', 'null',
            '-'
        ]
        
        try:
            _, stderr = await self.ffmpeg.run_command(cmd)
            
            # Extract complexity values (YMIN, YMAX, etc.)
            complexity_values = []
            
            for line in stderr.split('\n'):
                if 'lavfi.signalstats.YDIF' in line:
                    match = re.search(r'lavfi\.signalstats\.YDIF=(\d+\.?\d*)', line)
                    if match:
                        complexity_values.append(float(match.group(1)))
            
            if complexity_values:
                return np.mean(complexity_values) / 100.0
            else:
                return 0.15  # Default average value
                
        except Exception:
            return 0.15
    
    async def _detect_film_content(self) -> bool:
        """Detect if content is film (24fps) telecined to 50i"""
        cmd = [
            'ffmpeg',
            '-i', str(self.video_path),
            '-t', str(self.sample_duration),
            '-vf', 'pullup,idet',
            '-f', 'null',
            '-'
        ]
        
        try:
            _, stderr = await self.ffmpeg.run_command(cmd)
            
            # Look for telecine patterns (3:2 pulldown adapted to PAL)
            repeated_frames = 0
            
            for line in stderr.split('\n'):
                if 'repeated' in line.lower():
                    repeated_frames += 1
            
            # If more than 20% repeated frames, likely film content
            total_frames = self.sample_duration * 25
            return (repeated_frames / max(total_frames, 1)) > 0.2
            
        except Exception:
            return False
    
    async def _analyze_temporal_consistency(self) -> float32:
        """Analyze temporal consistency between frames"""
        cmd = [
            'ffmpeg',
            '-i', str(self.video_path),
            '-t', str(self.sample_duration),
            '-vf', 'tblend=all_mode=difference,metadata=print:file=-',
            '-f', 'null',
            '-'
        ]
        
        try:
            _, stderr = await self.ffmpeg.run_command(cmd)
            
            # Measure difference between consecutive frames
            differences = []
            
            for line in stderr.split('\n'):
                if 'lavfi.tblend' in line:
                    match = re.search(r'(\d+\.?\d*)', line)
                    if match:
                        differences.append(float(match.group(1)))
            
            if differences:
                # Low variance = good temporal consistency
                return 1.0 - min(np.var(differences) / 1000.0, 1.0)
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    # Optimized scoring functions with Cython
    @cython.ccall
    @cython.returns(float32)
    def _score_yadif(self, a: VideoAnalysis) -> float32:
        """Score for yadif (standard filter, versatile)"""
        cdef float32 score = 0.5  # Base
        
        # Good for low to medium motion
        if a.avg_motion < self.THRESHOLDS['high_motion']:
            score += 0.2
        
        # Good for low to medium complexity
        if a.complexity < self.THRESHOLDS['high_complexity']:
            score += 0.15
        
        # Penalty for very high motion
        if a.avg_motion > self.THRESHOLDS['high_motion'] * 1.5:
            score -= 0.2
        
        # Bonus if clearly interlaced
        if a.interlaced_frames > 0.7:
            score += 0.15
        
        return max(0.0, min(1.0, score))
    
    @cython.ccall
    @cython.returns(float32)
    def _score_bwdif(self, a: VideoAnalysis) -> float32:
        """Score for bwdif (better quality than yadif)"""
        cdef float32 score = 0.6  # Higher base
        
        # Excellent for medium to high complexity
        if a.complexity > 0.15:
            score += 0.25
        
        # Good for medium motion
        if 0.05 < a.avg_motion < 0.2:
            score += 0.2
        
        # Bonus for lots of detail
        if a.complexity > self.THRESHOLDS['high_complexity']:
            score += 0.15
        
        # Slight penalty for very fast motion
        if a.avg_motion > 0.25:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    @cython.ccall
    @cython.returns(float32)
    def _score_estdif(self, a: VideoAnalysis) -> float32:
        """Score for estdif (optimized for fast motion)"""
        cdef float32 score = 0.4  # Base
        
        # Excellent for high motion
        if a.avg_motion > self.THRESHOLDS['high_motion']:
            score += 0.4
        
        # Bonus for very fast motion
        if a.avg_motion > 0.2:
            score += 0.2
        
        # Good for many scene changes
        scene_rate = a.scene_changes / (self.sample_duration * 25)
        if scene_rate > self.THRESHOLDS['scene_change_rate']:
            score += 0.15
        
        # Penalty if low motion
        if a.avg_motion < self.THRESHOLDS['low_motion']:
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    @cython.ccall
    @cython.returns(float32)
    def _score_w3fdif(self, a: VideoAnalysis) -> float32:
        """Score for w3fdif (good for film content)"""
        cdef float32 score = 0.45  # Base
        
        # Excellent for film content
        if a.has_film_content:
            score += 0.4
        
        # Good for high temporal consistency
        if a.temporal_consistency > 0.7:
            score += 0.25
        
        # Bonus if low motion (typical for film)
        if a.avg_motion < self.THRESHOLDS['low_motion']:
            score += 0.15
        
        # Penalty if high motion variance
        if a.motion_variance > 0.1:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    @cython.ccall
    @cython.returns(float32)
    def _score_nnedi(self, a: VideoAnalysis) -> float32:
        """Score for nnedi (neural network, high quality)"""
        cdef float32 score = 0.55  # High base (superior quality)
        
        # Excellent for high quality and fine details
        if a.complexity > 0.2:
            score += 0.3
        
        # Very good for low to medium motion
        if a.avg_motion < self.THRESHOLDS['high_motion']:
            score += 0.2
        
        # Bonus for high temporal consistency
        if a.temporal_consistency > 0.6:
            score += 0.15
        
        # Penalty if very fast motion (computationally expensive)
        if a.avg_motion > 0.25:
            score -= 0.25
        
        # Bonus if very detailed content
        if a.complexity > self.THRESHOLDS['high_complexity']:
            score += 0.2
        
        return max(0.0, min(1.0, score))
    
    async def recommend_filter(self) -> FilterRecommendation:
        """
        Recommend the best filter based on analysis
        
        Decision logic:
        - YADIF: Simple content, low to medium motion, good compromise
        - BWDIF: Complex content, medium motion, better quality
        - ESTDIF: High motion, sports, action
        - W3FDIF: Film content, high temporal consistency
        - NNEDI: High quality, fine details, low to medium motion
        """
        if self.analysis is None:
            await self.analyze_video()
        
        a = self.analysis
        t = self.THRESHOLDS
        
        print(f"\n📊 Analysis results:")
        print(f"  Average motion: {a.avg_motion:.3f}")
        print(f"  Max motion: {a.max_motion:.3f}")
        print(f"  Motion variance: {a.motion_variance:.3f}")
        print(f"  Scene changes: {a.scene_changes}")
        print(f"  Spatial complexity: {a.complexity:.3f}")
        print(f"  Interlacing ratio: {a.interlaced_frames:.3f}")
        print(f"  Film content: {a.has_film_content}")
        print(f"  Temporal consistency: {a.temporal_consistency:.3f}")
        
        # Calculate score for each filter
        scores = {
            DeinterlaceFilter.YADIF: self._score_yadif(a),
            DeinterlaceFilter.BWDIF: self._score_bwdif(a),
            DeinterlaceFilter.ESTDIF: self._score_estdif(a),
            DeinterlaceFilter.W3FDIF: self._score_w3fdif(a),
            DeinterlaceFilter.NNEDI: self._score_nnedi(a)
        }
        
        # Select the best
        best_filter = max(scores.items(), key=lambda x: x[1])
        
        # Find alternative
        scores_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        alternative = scores_sorted[1][0] if len(scores_sorted) > 1 else None
        
        # Determine mode (0=25fps, 1=50fps)
        mode = 1 if a.avg_motion > t['low_motion'] else 0
        
        # Generate reason
        reason = self._generate_reason(best_filter[0], a, t)
        
        return FilterRecommendation(
            filter=best_filter[0],
            mode=mode,
            confidence=best_filter[1],
            reason=reason,
            alternative=alternative
        )
    
    def _generate_reason(self, filter: DeinterlaceFilter, 
                        a: VideoAnalysis, t: Dict) -> str:
        """Generate explanation for recommendation"""
        reasons = []
        
        if filter == DeinterlaceFilter.YADIF:
            reasons.append("Versatile filter suited for content")
            if a.avg_motion < t['high_motion']:
                reasons.append("low to medium motion detected")
            if a.complexity < t['high_complexity']:
                reasons.append("moderate spatial complexity")
        
        elif filter == DeinterlaceFilter.BWDIF:
            reasons.append("Better quality than yadif recommended")
            if a.complexity > 0.15:
                reasons.append("high spatial complexity detected")
            reasons.append("optimal detail preservation")
        
        elif filter == DeinterlaceFilter.ESTDIF:
            reasons.append("Optimized for fast motion")
            if a.avg_motion > t['high_motion']:
                reasons.append(f"high motion detected ({a.avg_motion:.3f})")
            if a.scene_changes > t['scene_change_rate'] * self.sample_duration * 25:
                reasons.append("multiple scene changes")
        
        elif filter == DeinterlaceFilter.W3FDIF:
            reasons.append("Optimized for film content")
            if a.has_film_content:
                reasons.append("telecined content detected")
            if a.temporal_consistency > 0.7:
                reasons.append("excellent temporal consistency")
                
        elif filter == DeinterlaceFilter.NNEDI:
            reasons.append("Maximum quality with neural network")
            if a.complexity > 0.2:
                reasons.append("fine details and high complexity")
            if a.avg_motion < t['high_motion']:
                reasons.append("motion suitable for neural processing")
            reasons.append("exceptional edge preservation")
            
        return ", ".join(reasons)
    
    def generate_ffmpeg_command(self, output_path: str, 
                               codec: str = 'prores_ks',
                               profile: int = 3) -> str:
        """
        Generate complete FFmpeg command with recommended filter
        """
        if self.analysis is None:
            raise ValueError("Video must be analyzed first")
        
        rec = self.recommend_filter_sync()
        
        # Build filter
        if rec.filter == DeinterlaceFilter.YADIF:
            vf = f"yadif={rec.mode}"
        elif rec.filter == DeinterlaceFilter.BWDIF:
            vf = f"bwdif={rec.mode}"
        elif rec.filter == DeinterlaceFilter.ESTDIF:
            vf = f"estdif=mode={rec.mode}"
        elif rec.filter == DeinterlaceFilter.NNEDI:
            vf = f"nnedi=weights=nnedi3_weights.bin"
        else:  # W3FDIF
            vf = "w3fdif"
        
        # Add pixel format for ProRes
        if codec == 'prores_ks':
            vf += ",format=yuv422p10le"
        
        cmd = (
            f"ffmpeg -i {self.video_path} "
            f"-vf \"{vf}\" "
            f"-c:v {codec} -profile:v {profile} "
            f"-c:a copy "
            f"{output_path}"
        )
        
        return cmd
    
    def recommend_filter_sync(self) -> FilterRecommendation:
        """
        Synchronous version of recommend_filter
        For use in synchronous contexts
        """
        if self.analysis is None:
            raise ValueError("Video must be analyzed first")
        
        return asyncio.run(self.recommend_filter())
    
    async def close(self):
        """Cleanup resources"""
        self.ffmpeg.close()


async def process_video_async(video_path: str, output_path: str) -> FilterRecommendation:
    """
    Async video processing example
    """
    selector = DeinterlaceSelectorAsync(video_path, sample_duration=30)
    
    try:
        # Analyze
        print("=" * 60)
        await selector.analyze_video()
        print("=" * 60)
        
        # Get recommendation
        recommendation = await selector.recommend_filter()
        
        print(f"\n✅ Recommendation:")
        print(f"  Filter: {recommendation.filter.value}")
        print(f"  Mode: {recommendation.mode} ({'50fps' if recommendation.mode == 1 else '25fps'})")
        print(f"  Confidence: {recommendation.confidence:.1%}")
        print(f"  Reason: {recommendation.reason}")
        if recommendation.alternative:
            print(f"  Alternative: {recommendation.alternative.value}")
        
        # Generate command
        print(f"\n🎬 FFmpeg command:")
        cmd = selector.generate_ffmpeg_command(output_path)
        print(f"  {cmd}")
        
        return recommendation
        
    finally:
        await selector.close()


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python deinterlace_selector_async.py <video_file> [output_file]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output.mov"
    
    # Run async processing
    asyncio.run(process_video_async(video_path, output_path))


if __name__ == '__main__':
    main()