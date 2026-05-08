# SSDP Sample Dataset

50 representative examples from the Egyptian Arabic synthetic speech dataset.

## Contents
- `sample_manifest.jsonl`: Metadata (text, duration, filepath)
- `audio/`: 50 WAV files (16kHz, mono)

## Usage

```python
import json
import soundfile as sf

with open('sample_manifest.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        sample = json.loads(line)
        audio, sr = sf.read(sample['audio_filepath'])
        print(f"Text: {sample['text']}")
        print(f"Duration: {sample['duration']:.2f}s")
```

## Quality
- All samples passed automated quality filters
- Average duration: 4.2 seconds
- Approval rate: 84% from original synthesis