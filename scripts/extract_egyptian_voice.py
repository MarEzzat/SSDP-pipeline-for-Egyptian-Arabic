from datasets import load_dataset
import librosa
import soundfile as sf
import os
import numpy as np

def extract_egyptian_voice():
    print(" Loading Common Voice 18 Arabic dataset...")
    print("   (This will download ~3GB on first run, cached afterward)\n")
    
    # Load validated split (best quality)
    dataset = load_dataset(
        "MohamedRashad/common-voice-18-arabic", 
        split="validation",  # Use validation for quality
        streaming=True  # Don't download everything
    )
    
    print(" Dataset loaded in streaming mode")
    print("\n Searching for Egyptian Arabic samples...\n")
    
    # Search for Egyptian samples
    egyptian_samples = []
    
    for i, sample in enumerate(dataset):
        if i > 500:  # Check first 500 samples
            break
            
        # Check if accent field contains "Egypt" or similar
        accent = sample.get('accent', '').lower()
        
        if any(keyword in accent for keyword in ['egypt', 'egyptian', 'مصر', 'cairo']):
            # Get audio info
            audio = sample['audio']
            audio_array = audio['array']
            sample_rate = audio['sampling_rate']
            duration = len(audio_array) / sample_rate
            
            # Filter: 3-15 seconds, good quality
            if 3.0 <= duration <= 15.0:
                # Check audio quality (not too quiet)
                rms = np.sqrt(np.mean(audio_array**2))
                if rms > 0.01:  # Not silent
                    egyptian_samples.append({
                        'audio': audio_array,
                        'sr': sample_rate,
                        'text': sample['sentence'],
                        'duration': duration,
                        'accent': accent,
                        'gender': sample.get('gender', 'unknown'),
                        'rms': rms
                    })
                    
                    print(f" Found: {accent} | {duration:.1f}s | {sample['sentence'][:50]}...")
                    
                    if len(egyptian_samples) >= 5:
                        break
    
    # Fallback if no Egyptian found
    if len(egyptian_samples) == 0:
        dataset = load_dataset(
            "MohamedRashad/common-voice-18-arabic", 
            split="validation",
            streaming=True
        )
        
        for i, sample in enumerate(dataset):
            if i >= 10:
                break
            
            audio = sample['audio']
            audio_array = audio['array']
            sample_rate = audio['sampling_rate']
            duration = len(audio_array) / sample_rate
            
            if 3.0 <= duration <= 15.0:
                egyptian_samples.append({
                    'audio': audio_array,
                    'sr': sample_rate,
                    'text': sample['sentence'],
                    'duration': duration,
                    'accent': 'generic_arabic',
                    'gender': sample.get('gender', 'unknown'),
                    'rms': np.sqrt(np.mean(audio_array**2))
                })
    
    # Pick best sample (longest with good volume)
    best_sample = max(egyptian_samples, key=lambda x: x['duration'] * x['rms'])
    
    print(f"\n Selected sample:")
    print(f"   Accent: {best_sample['accent']}")
    print(f"   Duration: {best_sample['duration']:.2f}s")
    print(f"   Text: {best_sample['text']}")
    print(f"   Gender: {best_sample['gender']}")
    
    # Resample to 16kHz mono for TTS
    audio_16k = librosa.resample(
        best_sample['audio'], 
        orig_sr=best_sample['sr'], 
        target_sr=16000
    )
    
    # Normalize audio
    audio_16k = audio_16k / np.max(np.abs(audio_16k)) * 0.95
    
    # Save
    os.makedirs('assets', exist_ok=True)
    output_path = 'assets/egyptian_speaker.wav'
    
    sf.write(output_path, audio_16k, 16000)
    
    print(f"\n Saved: {output_path}")
    print(f"   Format: 16kHz, mono, WAV")
    print(f"   Size: {os.path.getsize(output_path) / 1024:.1f} KB")
    
    # Also save metadata
    import json
    metadata = {
        'source': 'Mozilla Common Voice 18 Arabic',
        'accent': best_sample['accent'],
        'text': best_sample['text'],
        'duration': best_sample['duration'],
        'gender': best_sample['gender']
    }
    
    with open('assets/speaker_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"OKK Metadata saved: assets/speaker_metadata.json")
    
    return output_path

if __name__ == "__main__":
    print("=" * 70)
    print("SSDP: Egyptian Arabic Reference Voice Extractor")
    print("=" * 70)
    print()
    
    try:
        output = extract_egyptian_voice()
        print("\n" + "=" * 70)
        print(" SUCCESS! You can now run: python src/2_synthesize_audio.py")
        print("=" * 70)
    except Exception as e:
        print(f"\n ERROR: {e}")
        print("\nFallback: Record yourself saying:")
        print('   "أهلاً، إزيك؟ أنا سعيد إني أتكلم معاك النهاردة"')
        print("   Save as: assets/egyptian_speaker.wav (16kHz, mono)")