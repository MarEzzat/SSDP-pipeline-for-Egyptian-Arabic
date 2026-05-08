import torch
from TTS.api import TTS
import json
import os
from tqdm import tqdm
import yaml
import logging
import soundfile as sf

class TTSOrchestrator:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Coqui XTTSv2
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        
        self.checkpoint_file = os.path.join(
            config['pipeline']['checkpoint_dir'], 
            'synthesis_checkpoint.json'
        )
        self.results_file = "data/synthesis_results.jsonl"
        
    def _get_duration(self, audio_path):
        try:
            info = sf.info(audio_path)
            return info.duration
        except:
            return 0.0

    def _get_flags(self, duration, text):
        flags = []
        if duration < 1.0:
            flags.append("too_short")
        if duration > 20.0:
            flags.append("too_long")
        if len(text) > 200:
            flags.append("long_text")
        return flags

    def synthesize_batch(self, prompts):
        """Synthesize audio with checkpointing and result logging"""
        completed = self._load_checkpoint()

        # Load already-written result IDs to avoid duplicates
        written_ids = set()
        if os.path.exists(self.results_file):
            with open(self.results_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        written_ids.add(json.loads(line)["id"])
                    except:
                        pass

        results_f = open(self.results_file, "a", encoding="utf-8")

        try:
            for idx, prompt in enumerate(tqdm(prompts)):
                if prompt['id'] in completed:
                    continue
                    
                try:
                    audio_path = os.path.join(
                        self.config['tts']['output_dir'],
                        f"{prompt['id']}.wav"
                    )
                    
                    # Synthesize
                    self.tts.tts_to_file(
                        text=prompt['text'],
                        file_path=audio_path,
                        speaker_wav=self.config['tts']['speaker_wav'],
                        language="ar"
                    )
                    
                    # Get audio metadata
                    duration = self._get_duration(audio_path)
                    flags = self._get_flags(duration, prompt['text'])

                    # Write result to jsonl
                    if prompt['id'] not in written_ids:
                        result = {
                            "id": prompt['id'],
                            "text": prompt['text'],
                            "audio_path": audio_path,
                            "duration": duration,
                            "flags": flags
                        }
                        results_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        results_f.flush()
                        written_ids.add(prompt['id'])

                    # Log success
                    completed.add(prompt['id'])
                    
                    # Checkpoint every 10 samples
                    if idx % 10 == 0:
                        self._save_checkpoint(completed)
                        
                except Exception as e:
                    logging.error(f"Failed {prompt['id']}: {e}")
                    continue
        finally:
            results_f.close()
        
        self._save_checkpoint(completed)

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return set(json.load(f))
        return set()

    def _save_checkpoint(self, completed):
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        with open(self.checkpoint_file, 'w') as f:
            json.dump(list(completed), f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load config
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load prompts
    prompts = []
    with open("data/prompts.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))

    print(f"Loaded {len(prompts)} prompts")

    # Make sure output dirs exist
    os.makedirs(config['tts']['output_dir'], exist_ok=True)
    os.makedirs(config['pipeline']['checkpoint_dir'], exist_ok=True)

    # Run
    orchestrator = TTSOrchestrator(config)
    orchestrator.synthesize_batch(prompts)

    # Summary
    count = sum(1 for _ in open("data/synthesis_results.jsonl", encoding="utf-8"))
    print(f"Done! {count} results written to data/synthesis_results.jsonl")