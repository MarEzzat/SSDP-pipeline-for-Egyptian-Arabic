import librosa
import numpy as np

def auto_quality_check(audio_path, text, config):
    """Return quality signals"""
    y, sr = librosa.load(audio_path, sr=16000)
    
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Rule-based filters
    flags = []
    
    # 1. Duration sanity (3-5 chars per second in Arabic)
    expected_duration = len(text) / 4.0  # ~4 chars/sec
    if duration < config['review']['auto_reject_min_duration']:
        flags.append("TOO_SHORT")
    elif duration > config['review']['auto_reject_max_duration']:
        flags.append("TOO_LONG")
    elif abs(duration - expected_duration) > expected_duration * 0.5:
        flags.append("DURATION_MISMATCH")
    
    # 2. Audio quality
    rms = librosa.feature.rms(y=y)[0]
    if np.mean(rms) < 0.01:
        flags.append("LOW_VOLUME")
    
    # 3. Clipping detection
    if np.max(np.abs(y)) > 0.99:
        flags.append("CLIPPING")
    
    return {
        "duration": duration,
        "flags": flags,
        "auto_reject": len(flags) > 0
    }