"""
Cython-optimized core for deinterlace filter selection.
"""
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as np
from libc.math cimport fabs, sqrt
from typing import Dict
import re
from enum import Enum

# Type definitions
ctypedef double float64_t
ctypedef float float32_t
ctypedef long long int64_t
ctypedef int int32_t

np.import_array()

class DeinterlaceFilter(Enum):
    """Available deinterlacing filters"""
    YADIF = "yadif"
    BWDIF = "bwdif"
    ESTDIF = "estdif"
    W3FDIF = "w3fdif"
    NNEDI = "nnedi"


cdef class VideoAnalysis:
    """Video analysis results with Cython optimization"""
    cdef public float64_t avg_motion
    cdef public float64_t max_motion
    cdef public float64_t motion_variance
    cdef public int32_t scene_changes
    cdef public float64_t complexity
    cdef public float64_t interlaced_frames
    cdef public bint has_film_content
    cdef public float64_t temporal_consistency
    
    def __init__(self, 
                 float64_t avg_motion,
                 float64_t max_motion,
                 float64_t motion_variance,
                 int32_t scene_changes,
                 float64_t complexity,
                 float64_t interlaced_frames,
                 bint has_film_content,
                 float64_t temporal_consistency):
        self.avg_motion = avg_motion
        self.max_motion = max_motion
        self.motion_variance = motion_variance
        self.scene_changes = scene_changes
        self.complexity = complexity
        self.interlaced_frames = interlaced_frames
        self.has_film_content = has_film_content
        self.temporal_consistency = temporal_consistency
    
    def __repr__(self):
        return (f"VideoAnalysis(avg_motion={self.avg_motion:.3f}, "
                f"max_motion={self.max_motion:.3f}, "
                f"complexity={self.complexity:.3f})")


cdef class FilterScorer:
    """Cython-optimized filter scoring"""
    cdef dict THRESHOLDS
    
    def __init__(self):
        self.THRESHOLDS = {
            'high_motion': 0.15,
            'low_motion': 0.05,
            'high_complexity': 0.3,
            'scene_change_rate': 0.05,
        }
    
    cpdef float64_t score_yadif(self, VideoAnalysis analysis):
        """Score for yadif (standard filter, versatile)"""
        cdef float64_t score = 0.5
        
        # Good for low to medium motion
        if analysis.avg_motion < self.THRESHOLDS['high_motion']:
            score += 0.2
        
        # Good for low to medium complexity
        if analysis.complexity < self.THRESHOLDS['high_complexity']:
            score += 0.15
        
        # Penalty for very high motion
        if analysis.avg_motion > self.THRESHOLDS['high_motion'] * 1.5:
            score -= 0.2
        
        # Bonus if clearly interlaced
        if analysis.interlaced_frames > 0.7:
            score += 0.15
        
        return max(0.0, min(1.0, score))
    
    cpdef float64_t score_bwdif(self, VideoAnalysis analysis):
        """Score for bwdif (better quality than yadif)"""
        cdef float64_t score = 0.6
        
        # Excellent for medium to high complexity
        if analysis.complexity > 0.15:
            score += 0.25
        
        # Good for medium motion
        if 0.05 < analysis.avg_motion < 0.2:
            score += 0.2
        
        # Bonus for lots of detail
        if analysis.complexity > self.THRESHOLDS['high_complexity']:
            score += 0.15
        
        # Slight penalty for very fast motion
        if analysis.avg_motion > 0.25:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    cpdef float64_t score_estdif(self, VideoAnalysis analysis):
        """Score for estdif (optimized for fast motion)"""
        cdef float64_t score = 0.4
        cdef float64_t scene_rate
        
        # Excellent for high motion
        if analysis.avg_motion > self.THRESHOLDS['high_motion']:
            score += 0.4
        
        # Bonus for very fast motion
        if analysis.avg_motion > 0.2:
            score += 0.2
        
        # Good for many scene changes
        scene_rate = analysis.scene_changes / 750.0  # Assuming 30s * 25fps
        if scene_rate > self.THRESHOLDS['scene_change_rate']:
            score += 0.15
        
        # Penalty if low motion
        if analysis.avg_motion < self.THRESHOLDS['low_motion']:
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    cpdef float64_t score_w3fdif(self, VideoAnalysis analysis):
        """Score for w3fdif (good for film content)"""
        cdef float64_t score = 0.45
        
        # Excellent for film content
        if analysis.has_film_content:
            score += 0.4
        
        # Good for high temporal consistency
        if analysis.temporal_consistency > 0.7:
            score += 0.25
        
        # Bonus if low motion (typical for film)
        if analysis.avg_motion < self.THRESHOLDS['low_motion']:
            score += 0.15
        
        # Penalty if high motion variance
        if analysis.motion_variance > 0.1:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    cpdef float64_t score_nnedi(self, VideoAnalysis analysis):
        """Score for nnedi (neural network, high quality)"""
        cdef float64_t score = 0.55
        
        # Excellent for high quality and fine details
        if analysis.complexity > 0.2:
            score += 0.3
        
        # Very good for low to medium motion
        if analysis.avg_motion < self.THRESHOLDS['high_motion']:
            score += 0.2
        
        # Bonus for high temporal consistency
        if analysis.temporal_consistency > 0.6:
            score += 0.15
        
        # Penalty if very fast motion (computationally expensive)
        if analysis.avg_motion > 0.25:
            score -= 0.25
        
        # Bonus if very detailed content
        if analysis.complexity > self.THRESHOLDS['high_complexity']:
            score += 0.2
        
        return max(0.0, min(1.0, score))


cdef class AnalysisProcessor:
    """Cython-optimized video analysis processor"""
    
    @staticmethod
    def process_motion_data(motion_values_list):
        """Process motion data with numpy array operations"""
        if not motion_values_list:
            return {'avg': 0.1, 'max': 0.2, 'variance': 0.05, 'scene_changes': 5}
        
        cdef np.ndarray motion_array = np.array(motion_values_list, dtype=np.float64)
        cdef float64_t avg = np.mean(motion_array) / 10000.0
        cdef float64_t max_val = np.max(motion_array) / 10000.0
        cdef float64_t variance = np.var(motion_array) / 100000000.0
        
        return {
            'avg': avg,
            'max': max_val,
            'variance': variance,
            'scene_changes': len([x for x in motion_values_list if x > 50000])
        }
    
    @staticmethod
    def parse_interlace_data(stderr: str):
        """Parse interlacing detection results"""
        cdef int tff = 0, bff = 0, progressive = 0
        
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
        
        cdef int total = tff + bff + progressive
        if total == 0:
            return {'interlaced_ratio': 0.5}
        
        cdef float64_t interlaced_ratio = (tff + bff) / total
        
        return {
            'interlaced_ratio': interlaced_ratio,
            'tff': tff,
            'bff': bff,
            'progressive': progressive
        }
    
    @staticmethod
    def parse_complexity_data(stderr: str):
        """Parse complexity data from signalstats"""
        cdef list complexity_values = []
        
        for line in stderr.split('\n'):
            if 'lavfi.signalstats.YDIF' in line:
                match = re.search(r'lavfi\.signalstats\.YDIF=(\d+\.?\d*)', line)
                if match:
                    complexity_values.append(float(match.group(1)))
        
        if complexity_values:
            return float(np.mean(complexity_values) / 100.0)
        else:
            return 0.15


# Pure Python functions (wrappers for Cython classes)
def create_video_analysis(
    avg_motion: float,
    max_motion: float,
    motion_variance: float,
    scene_changes: int,
    complexity: float,
    interlaced_frames: float,
    has_film_content: bool,
    temporal_consistency: float
) -> VideoAnalysis:
    """Create a VideoAnalysis instance"""
    return VideoAnalysis(
        avg_motion, max_motion, motion_variance,
        scene_changes, complexity, interlaced_frames,
        has_film_content, temporal_consistency
    )

# Add this class definition near the top (after DeinterlaceFilter)
class FilterRecommendation:
    """Filter recommendation with justification"""
    def __init__(self, filter, mode, confidence, reason, alternative=None):
        self.filter = filter
        self.mode = mode
        self.confidence = confidence
        self.reason = reason
        self.alternative = alternative
    
    def __repr__(self):
        return f"FilterRecommendation(filter={self.filter.value}, confidence={self.confidence:.1%})"


def recommend_filter(analysis: VideoAnalysis) -> FilterRecommendation:
    """Get filter recommendation based on analysis"""
    cdef FilterScorer scorer = FilterScorer()
    
    scores = {
        DeinterlaceFilter.YADIF: scorer.score_yadif(analysis),
        DeinterlaceFilter.BWDIF: scorer.score_bwdif(analysis),
        DeinterlaceFilter.ESTDIF: scorer.score_estdif(analysis),
        DeinterlaceFilter.W3FDIF: scorer.score_w3fdif(analysis),
        DeinterlaceFilter.NNEDI: scorer.score_nnedi(analysis)
    }
    
    # Find best filter
    best_filter = max(scores.items(), key=lambda x: x[1])
    
    # Find alternative
    scores_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    alternative = scores_sorted[1][0] if len(scores_sorted) > 1 else None
    
    # Determine mode (0=25fps, 1=50fps)
    cdef int mode = 1 if analysis.avg_motion > 0.05 else 0
    
    # Generate reason
    cdef str reason = ""
    if best_filter[0] == DeinterlaceFilter.YADIF:
        reason = "Versatile filter for general content"
        if analysis.avg_motion < 0.15:
            reason += ", low to medium motion"
        if analysis.complexity < 0.3:
            reason += ", moderate complexity"
    
    elif best_filter[0] == DeinterlaceFilter.BWDIF:
        reason = "Better quality for complex scenes"
        if analysis.complexity > 0.15:
            reason += ", high spatial complexity"
    
    elif best_filter[0] == DeinterlaceFilter.ESTDIF:
        reason = "Optimized for fast motion"
        if analysis.avg_motion > 0.15:
            reason += f", high motion ({analysis.avg_motion:.3f})"
    
    elif best_filter[0] == DeinterlaceFilter.W3FDIF:
        reason = "Best for film content"
        if analysis.has_film_content:
            reason += ", telecined content detected"
    
    elif best_filter[0] == DeinterlaceFilter.NNEDI:
        reason = "Highest quality with neural processing"
        if analysis.complexity > 0.2:
            reason += ", fine details present"
    
    # Return FilterRecommendation object
    return FilterRecommendation(
        filter=best_filter[0],
        mode=mode,
        confidence=best_filter[1],
        reason=reason,
        alternative=alternative
    )