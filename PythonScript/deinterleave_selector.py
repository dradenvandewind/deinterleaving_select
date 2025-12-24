#!/usr/bin/env python3
"""
Algorithme de sélection automatique du meilleur filtre de désentrelacement
basé sur l'analyse du mouvement et des caractéristiques de la vidéo.
"""

import subprocess
import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class DeinterlaceFilter(Enum):
    """Filtres de désentrelacement disponibles"""
    YADIF = "yadif"
    BWDIF = "bwdif"
    ESTDIF = "estdif"
    W3FDIF = "w3fdif"
    NNEDI = "nnedi" 


@dataclass
class VideoAnalysis:
    """Résultats de l'analyse vidéo"""
    avg_motion: float
    max_motion: float
    motion_variance: float
    scene_changes: int
    complexity: float
    interlaced_frames: float
    has_film_content: bool
    temporal_consistency: float


@dataclass
class FilterRecommendation:
    """Recommandation de filtre avec justification"""
    filter: DeinterlaceFilter
    mode: int
    confidence: float
    reason: str
    alternative: Optional[DeinterlaceFilter] = None


class DeinterlaceSelector:
    """
    Sélecteur intelligent de filtre de désentrelacement
    basé sur l'analyse du contenu vidéo
    """
    
    # Seuils de décision
    THRESHOLDS = {
        'high_motion': 0.15,      # Motion élevé
        'low_motion': 0.05,        # Motion faible
        'high_complexity': 0.3,    # Complexité spatiale élevée
        'scene_change_rate': 0.05, # Taux de changement de scène
        'film_confidence': 0.7,    # Confiance détection film
    }
    
    def __init__(self, video_path: str, sample_duration: int = 30):
        """
        Args:
            video_path: Chemin vers la vidéo à analyser
            sample_duration: Durée d'échantillon pour l'analyse (secondes)
        """
        self.video_path = Path(video_path)
        self.sample_duration = sample_duration
        self.analysis: Optional[VideoAnalysis] = None
    
    def analyze_video(self) -> VideoAnalysis:
        """
        Première passe: analyse du mouvement et des caractéristiques
        """
        print(f"🔍 Analyse de la vidéo: {self.video_path.name}")
        
        # 1. Analyse du mouvement avec freezedetect et idet
        motion_data = self._analyze_motion()
        
        # 2. Détection d'entrelacement
        interlace_data = self._detect_interlacing()
        
        # 3. Analyse de complexité spatiale
        complexity = self._analyze_complexity()
        
        # 4. Détection de contenu film (telecine)
        film_detection = self._detect_film_content()
        
        # 5. Analyse de cohérence temporelle
        temporal = self._analyze_temporal_consistency()
        
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
    
    def _analyze_motion(self) -> Dict[str, float]:
        """Analyse le mouvement avec le filtre mestimate"""
        cmd = [
            'ffmpeg',
            '-i', str(self.video_path),
            '-t', str(self.sample_duration),
            '-vf', 'mestimate=method=esa,metadata=print:file=-',
            '-f', 'null',
            '-'
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parser les métadonnées de mouvement
            motion_values = []
            scene_changes = 0
            
            for line in result.stderr.split('\n'):
                # Rechercher les valeurs de mouvement
                if 'lavfi.mestimate.mb_sad' in line:
                    match = re.search(r'lavfi\.mestimate\.mb_sad=(\d+)', line)
                    if match:
                        motion_values.append(int(match.group(1)))
                
                # Détecter les changements de scène
                if 'lavfi.scene' in line or motion_values and motion_values[-1] > 50000:
                    scene_changes += 1
            
            if not motion_values:
                # Fallback: utiliser select et setpts
                return self._analyze_motion_fallback()
            
            motion_array = np.array(motion_values)
            
            return {
                'avg': np.mean(motion_array) / 10000.0,  # Normaliser
                'max': np.max(motion_array) / 10000.0,
                'variance': np.var(motion_array) / 100000000.0,
                'scene_changes': scene_changes
            }
            
        except subprocess.CalledProcessError:
            return self._analyze_motion_fallback()
    
    def _analyze_motion_fallback(self) -> Dict[str, float]:
        """Méthode alternative d'analyse du mouvement"""
        # Utiliser mpdecimate pour estimer le mouvement
        cmd = [
            'ffmpeg',
            '-i', str(self.video_path),
            '-t', str(self.sample_duration),
            '-vf', 'mpdecimate,metadata=print:file=-',
            '-f', 'null',
            '-'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Compter les frames dupliquées (peu de mouvement)
            dropped = result.stderr.count('drop')
            total_frames = self.sample_duration * 25  # Assumer 25fps
            
            motion_estimate = 1.0 - (dropped / max(total_frames, 1))
            
            return {
                'avg': motion_estimate * 0.2,
                'max': motion_estimate * 0.4,
                'variance': 0.05,
                'scene_changes': max(1, int(total_frames / 100))
            }
        except:
            # Valeurs par défaut conservatrices
            return {
                'avg': 0.1,
                'max': 0.2,
                'variance': 0.05,
                'scene_changes': 5
            }
    
    def _detect_interlacing(self) -> Dict[str, float]:
        """Détecte le taux d'entrelacement avec idet"""
        cmd = [
            'ffmpeg',
            '-i', str(self.video_path),
            '-t', str(self.sample_duration),
            '-vf', 'idet',
            '-f', 'null',
            '-'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parser les résultats idet
            tff = bff = progressive = 0
            
            for line in result.stderr.split('\n'):
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
                return {'interlaced_ratio': 0.5}  # Inconnu, assumer entrelacé
            
            interlaced_ratio = (tff + bff) / total
            
            return {
                'interlaced_ratio': interlaced_ratio,
                'tff': tff,
                'bff': bff,
                'progressive': progressive
            }
            
        except subprocess.CalledProcessError:
            return {'interlaced_ratio': 0.5}
    
    def _analyze_complexity(self) -> float:
        """Analyse la complexité spatiale de l'image"""
        cmd = [
            'ffmpeg',
            '-i', str(self.video_path),
            '-t', str(self.sample_duration),
            '-vf', 'select=not(mod(n\\,25)),signalstats',
            '-f', 'null',
            '-'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Extraire les valeurs de complexité (YMIN, YMAX, etc.)
            complexity_values = []
            
            for line in result.stderr.split('\n'):
                if 'lavfi.signalstats.YDIF' in line:
                    match = re.search(r'lavfi\.signalstats\.YDIF=(\d+\.?\d*)', line)
                    if match:
                        complexity_values.append(float(match.group(1)))
            
            if complexity_values:
                return np.mean(complexity_values) / 100.0
            else:
                return 0.15  # Valeur moyenne par défaut
                
        except subprocess.CalledProcessError:
            return 0.15
    
    def _detect_film_content(self) -> bool:
        """Détecte si le contenu est du film (24fps) téléciné en 50i"""
        cmd = [
            'ffmpeg',
            '-i', str(self.video_path),
            '-t', str(self.sample_duration),
            '-vf', 'pullup,idet',
            '-f', 'null',
            '-'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Chercher des patterns de telecine (3:2 pulldown adapté au PAL)
            repeated_frames = 0
            
            for line in result.stderr.split('\n'):
                if 'repeated' in line.lower():
                    repeated_frames += 1
            
            # Si plus de 20% de frames répétées, c'est probablement du film
            total_frames = self.sample_duration * 25
            return (repeated_frames / max(total_frames, 1)) > 0.2
            
        except subprocess.CalledProcessError:
            return False
    
    def _analyze_temporal_consistency(self) -> float:
        """Analyse la cohérence temporelle entre frames"""
        cmd = [
            'ffmpeg',
            '-i', str(self.video_path),
            '-t', str(self.sample_duration),
            '-vf', 'tblend=all_mode=difference,metadata=print:file=-',
            '-f', 'null',
            '-'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Mesurer la différence entre frames consécutives
            differences = []
            
            for line in result.stderr.split('\n'):
                if 'lavfi.tblend' in line:
                    match = re.search(r'(\d+\.?\d*)', line)
                    if match:
                        differences.append(float(match.group(1)))
            
            if differences:
                # Une variance faible = bonne cohérence temporelle
                return 1.0 - min(np.var(differences) / 1000.0, 1.0)
            else:
                return 0.5
                
        except subprocess.CalledProcessError:
            return 0.5
    def _score_nnedi(self, a: VideoAnalysis) -> float:
        """Score pour nnedi (neural network, haute qualité)"""
        score = 0.55  # Base élevée (qualité supérieure)
        
        # Excellent pour haute qualité et détails fins
        if a.complexity > 0.2:
            score += 0.3
        
        # Très bon pour mouvement faible à moyen
        if a.avg_motion < self.THRESHOLDS['high_motion']:
            score += 0.2
        
        # Bonus pour cohérence temporelle élevée
        if a.temporal_consistency > 0.6:
            score += 0.15
        
        # Pénalité si mouvement très rapide (coûteux en calcul)
        if a.avg_motion > 0.25:
            score -= 0.25
        
        # Bonus si contenu très détaillé
        if a.complexity > self.THRESHOLDS['high_complexity']:
            score += 0.2
        
        return max(0.0, min(1.0, score))
    
    def recommend_filter(self) -> FilterRecommendation:
        """
        Recommande le meilleur filtre basé sur l'analyse
        
        Logique de décision:
        - YADIF: Contenu simple, mouvement faible à moyen, bon compromis
        - BWDIF: Contenu complexe, mouvement moyen, meilleure qualité
        - ESTDIF: Mouvement élevé, sport, action
        - W3FDIF: Contenu film, cohérence temporelle élevée
        - NNEDI: Haute qualité, détails fins, mouvement faible à moyen
        """
        if self.analysis is None:
            self.analyze_video()
        
        a = self.analysis
        t = self.THRESHOLDS
        
        print(f"\n📊 Résultats de l'analyse:")
        print(f"  Mouvement moyen: {a.avg_motion:.3f}")
        print(f"  Mouvement max: {a.max_motion:.3f}")
        print(f"  Variance mouvement: {a.motion_variance:.3f}")
        print(f"  Changements de scène: {a.scene_changes}")
        print(f"  Complexité spatiale: {a.complexity:.3f}")
        print(f"  Ratio entrelacement: {a.interlaced_frames:.3f}")
        print(f"  Contenu film: {a.has_film_content}")
        print(f"  Cohérence temporelle: {a.temporal_consistency:.3f}")
        
        # Calcul du score pour chaque filtre
        scores = {
            DeinterlaceFilter.YADIF: self._score_yadif(a),
            DeinterlaceFilter.BWDIF: self._score_bwdif(a),
            DeinterlaceFilter.ESTDIF: self._score_estdif(a),
            DeinterlaceFilter.W3FDIF: self._score_w3fdif(a),
            DeinterlaceFilter.NNEDI: self._score_nnedi(a)
        }
        
        # Sélectionner le meilleur
        best_filter = max(scores.items(), key=lambda x: x[1])
        
        # Trouver l'alternative
        scores_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        alternative = scores_sorted[1][0] if len(scores_sorted) > 1 else None
        
        # Déterminer le mode (0=25fps, 1=50fps)
        mode = 1 if a.avg_motion > t['low_motion'] else 0
        
        # Générer la raison
        reason = self._generate_reason(best_filter[0], a, t)
        
        return FilterRecommendation(
            filter=best_filter[0],
            mode=mode,
            confidence=best_filter[1],
            reason=reason,
            alternative=alternative
        )
    
    def _score_yadif(self, a: VideoAnalysis) -> float:
        """Score pour yadif (filtre standard, polyvalent)"""
        score = 0.5  # Base
        
        # Bon pour mouvement faible à moyen
        if a.avg_motion < self.THRESHOLDS['high_motion']:
            score += 0.2
        
        # Bon pour complexité faible à moyenne
        if a.complexity < self.THRESHOLDS['high_complexity']:
            score += 0.15
        
        # Pénalité si mouvement très élevé
        if a.avg_motion > self.THRESHOLDS['high_motion'] * 1.5:
            score -= 0.2
        
        # Bonus si c'est clairement entrelacé
        if a.interlaced_frames > 0.7:
            score += 0.15
        
        return max(0.0, min(1.0, score))
    
    def _score_bwdif(self, a: VideoAnalysis) -> float:
        """Score pour bwdif (meilleure qualité que yadif)"""
        score = 0.6  # Base plus élevée
        
        # Excellent pour complexité moyenne à élevée
        if a.complexity > 0.15:
            score += 0.25
        
        # Bon pour mouvement moyen
        if 0.05 < a.avg_motion < 0.2:
            score += 0.2
        
        # Bonus si beaucoup de détails
        if a.complexity > self.THRESHOLDS['high_complexity']:
            score += 0.15
        
        # Légère pénalité si mouvement très rapide
        if a.avg_motion > 0.25:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _score_estdif(self, a: VideoAnalysis) -> float:
        """Score pour estdif (optimisé mouvement rapide)"""
        score = 0.4  # Base
        
        # Excellent pour mouvement élevé
        if a.avg_motion > self.THRESHOLDS['high_motion']:
            score += 0.4
        
        # Bonus pour mouvement très rapide
        if a.avg_motion > 0.2:
            score += 0.2
        
        # Bon pour beaucoup de changements de scène
        scene_rate = a.scene_changes / (self.sample_duration * 25)
        if scene_rate > self.THRESHOLDS['scene_change_rate']:
            score += 0.15
        
        # Pénalité si mouvement faible
        if a.avg_motion < self.THRESHOLDS['low_motion']:
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _score_w3fdif(self, a: VideoAnalysis) -> float:
        """Score pour w3fdif (bon pour contenu film)"""
        score = 0.45  # Base
        
        # Excellent pour contenu film
        if a.has_film_content:
            score += 0.4
        
        # Bon pour cohérence temporelle élevée
        if a.temporal_consistency > 0.7:
            score += 0.25
        
        # Bonus si mouvement faible (typique du film)
        if a.avg_motion < self.THRESHOLDS['low_motion']:
            score += 0.15
        
        # Pénalité si variance de mouvement élevée
        if a.motion_variance > 0.1:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _generate_reason(self, filter: DeinterlaceFilter, 
                        a: VideoAnalysis, t: Dict) -> str:
        """Génère une explication de la recommandation"""
        reasons = []
        
        if filter == DeinterlaceFilter.YADIF:
            reasons.append("Filtre polyvalent adapté au contenu")
            if a.avg_motion < t['high_motion']:
                reasons.append("mouvement faible à moyen détecté")
            if a.complexity < t['high_complexity']:
                reasons.append("complexité spatiale modérée")
        
        elif filter == DeinterlaceFilter.BWDIF:
            reasons.append("Meilleure qualité que yadif recommandée")
            if a.complexity > 0.15:
                reasons.append("complexité spatiale élevée détectée")
            reasons.append("préservation optimale des détails")
        
        elif filter == DeinterlaceFilter.ESTDIF:
            reasons.append("Optimisé pour mouvement rapide")
            if a.avg_motion > t['high_motion']:
                reasons.append(f"mouvement élevé détecté ({a.avg_motion:.3f})")
            if a.scene_changes > t['scene_change_rate'] * self.sample_duration * 25:
                reasons.append("nombreux changements de scène")
        
        elif filter == DeinterlaceFilter.W3FDIF:
            reasons.append("Optimisé pour contenu film")
            if a.has_film_content:
                reasons.append("contenu téléciné détecté")
            if a.temporal_consistency > 0.7:
                reasons.append("excellente cohérence temporelle")
                
        elif filter == DeinterlaceFilter.NNEDI:
            reasons.append("Qualité maximale avec réseau neuronal")
            if a.complexity > 0.2:
                reasons.append("détails fins et complexité élevée")
            if a.avg_motion < t['high_motion']:
                reasons.append("mouvement adapté pour traitement neuronal")
            reasons.append("préservation exceptionnelle des bords")
            
        return ", ".join(reasons)
    
    def generate_ffmpeg_command(self, output_path: str, 
                               codec: str = 'prores_ks',
                               profile: int = 3) -> str:
        """
        Génère la commande FFmpeg complète avec le filtre recommandé
        """
        if self.analysis is None:
            self.analyze_video()
        
        rec = self.recommend_filter()
        
        # Construire le filtre
        if rec.filter == DeinterlaceFilter.YADIF:
            vf = f"yadif={rec.mode}"
        elif rec.filter == DeinterlaceFilter.BWDIF:
            vf = f"bwdif={rec.mode}"
        elif rec.filter == DeinterlaceFilter.ESTDIF:
            vf = f"estdif=mode={rec.mode}"
        elif rec.filter == DeinterlaceFilter.NNEDI:
            # NNEDI avec paramètres optimaux
            #vf = f"nnedi=deint={'interlaced' if rec.mode == 1 else 'all'}"
            vf = f"nnedi=weights=nnedi3_weights.bin"
        else:  # W3FDIF
            vf = "w3fdif"
        
        
        # Ajouter le format de pixel pour ProRes
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


def main():
    """Exemple d'utilisation"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python deinterlace_selector.py <video_file> [output_file]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output.mov"
    
    # Créer le sélecteur
    selector = DeinterlaceSelector(video_path, sample_duration=30)
    
    # Analyser
    print("=" * 60)
    analysis = selector.analyze_video()
    print("=" * 60)
    
    # Obtenir la recommandation
    recommendation = selector.recommend_filter()
    
    print(f"\n✅ Recommandation:")
    print(f"  Filtre: {recommendation.filter.value}")
    print(f"  Mode: {recommendation.mode} ({'50fps' if recommendation.mode == 1 else '25fps'})")
    print(f"  Confiance: {recommendation.confidence:.1%}")
    print(f"  Raison: {recommendation.reason}")
    if recommendation.alternative:
        print(f"  Alternative: {recommendation.alternative.value}")
    
    # Générer la commande
    print(f"\n🎬 Commande FFmpeg:")
    cmd = selector.generate_ffmpeg_command(output_path)
    print(f"  {cmd}")
    
    print(f"\n💡 Pour exécuter:")
    print(f"  {cmd}")


if __name__ == '__main__':
    main()
