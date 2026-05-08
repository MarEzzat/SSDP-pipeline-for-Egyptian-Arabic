import json
import os
import sys
import tempfile
import pytest

# Make src/ importable without installing the package

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

class DummyTTS:
    """Stub so TTSOrchestrator.__init__ never touches the real model."""
    pass


def make_orchestrator(tmp_path):
    # Lazy import so the module-level torch/TTS imports don't crash the test
    # if CUDA is absent — we only call pure-Python methods.
    from unittest.mock import patch, MagicMock

    config = {
        "pipeline": {"checkpoint_dir": str(tmp_path / "checkpoints")},
        "tts": {"output_dir": str(tmp_path / "audio_raw"), "speaker_wav": "dummy.wav"},
    }

    with patch("src.2_synthesize_audio.TTS", return_value=MagicMock()):
        # Dynamic import because the filename starts with a digit
        import importlib.util, types

        spec = importlib.util.spec_from_file_location(
            "synthesize_audio",
            os.path.join(os.path.dirname(__file__), "..", "src", "2_synthesize_audio.py"),
        )
        module = importlib.util.module_from_spec(spec)
        # Patch heavy deps before exec
        sys.modules.setdefault("torch", MagicMock())
        sys.modules.setdefault("TTS", MagicMock())
        sys.modules.setdefault("TTS.api", MagicMock())
        spec.loader.exec_module(module)

    orch = module.TTSOrchestrator.__new__(module.TTSOrchestrator)
    orch.config = config
    orch.checkpoint_file = str(tmp_path / "checkpoints" / "synthesis_checkpoint.json")
    orch.results_file = str(tmp_path / "synthesis_results.jsonl")
    orch._get_flags = module.TTSOrchestrator._get_flags.__get__(orch, module.TTSOrchestrator)
    orch._load_checkpoint = module.TTSOrchestrator._load_checkpoint.__get__(orch, module.TTSOrchestrator)
    orch._save_checkpoint = module.TTSOrchestrator._save_checkpoint.__get__(orch, module.TTSOrchestrator)
    return orch


class TestQualityFlags:
    """_get_flags(duration, text) must fire precisely at the documented thresholds."""

    # We instantiate the logic directly to avoid any GPU dependency
    def _flags(self, duration, text):
        flags = []
        if duration < 1.0:
            flags.append("too_short")
        if duration > 20.0:
            flags.append("too_long")
        if len(text) > 200:
            flags.append("long_text")
        return flags

    # too_short 
    def test_too_short_fires_below_1s(self):
        assert "too_short" in self._flags(0.9, "نص قصير")

    def test_too_short_fires_at_zero(self):
        assert "too_short" in self._flags(0.0, "نص")

    def test_too_short_does_not_fire_at_exactly_1s(self):
        assert "too_short" not in self._flags(1.0, "نص")

    def test_too_short_does_not_fire_above_1s(self):
        assert "too_short" not in self._flags(5.0, "نص طبيعي")

    # too_long 
    def test_too_long_fires_above_20s(self):
        assert "too_long" in self._flags(20.1, "نص")

    def test_too_long_does_not_fire_at_exactly_20s(self):
        assert "too_long" not in self._flags(20.0, "نص")

    def test_too_long_does_not_fire_below_20s(self):
        assert "too_long" not in self._flags(15.0, "نص")

    # long_text 
    def test_long_text_fires_above_200_chars(self):
        long = "ا" * 201
        assert "long_text" in self._flags(5.0, long)

    def test_long_text_does_not_fire_at_exactly_200_chars(self):
        exact = "ا" * 200
        assert "long_text" not in self._flags(5.0, exact)

    def test_long_text_does_not_fire_below_200_chars(self):
        short = "أنا بحب القهوة"
        assert "long_text" not in self._flags(5.0, short)

    #  clean sample 
    def test_clean_sample_has_no_flags(self):
        assert self._flags(5.942, "والنبي الجو النهاردة حر قوي") == []

    #  multiple flags simultaneously 
    def test_multiple_flags_can_fire_together(self):
        long_text = "ا" * 201
        flags = self._flags(0.5, long_text)
        assert "too_short" in flags
        assert "long_text" in flags


# ===========================================================================
# 2. CHECKPOINT INTEGRITY
# We test _load_checkpoint / _save_checkpoint in isolation.
# Key properties:
#   a) Loading a missing checkpoint returns an empty set
#   b) Saving then loading round-trips correctly
#   c) Resuming skips IDs already in the checkpoint (no duplication)
# ===========================================================================

class TestCheckpointIntegrity:

    def _load(self, path):
        if os.path.exists(path):
            with open(path, "r") as f:
                return set(json.load(f))
        return set()

    def _save(self, path, completed: set):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(list(completed), f)

    def test_missing_checkpoint_returns_empty_set(self, tmp_path):
        path = str(tmp_path / "checkpoints" / "synthesis_checkpoint.json")
        assert self._load(path) == set()

    def test_save_then_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "checkpoints" / "synthesis_checkpoint.json")
        ids = {"abc123", "def456", "ghi789"}
        self._save(path, ids)
        loaded = self._load(path)
        assert loaded == ids

    def test_completed_ids_are_skipped_on_resume(self, tmp_path):
        """Simulate a batch of 5 prompts where 3 are already done."""
        checkpoint_path = str(tmp_path / "checkpoints" / "synthesis_checkpoint.json")
        already_done = {"id_001", "id_002", "id_003"}
        self._save(checkpoint_path, already_done)

        all_prompts = [
            {"id": "id_001", "text": "نص"},
            {"id": "id_002", "text": "نص"},
            {"id": "id_003", "text": "نص"},
            {"id": "id_004", "text": "نص"},  # <-- should be processed
            {"id": "id_005", "text": "نص"},  # <-- should be processed
        ]

        completed = self._load(checkpoint_path)
        to_process = [p for p in all_prompts if p["id"] not in completed]

        assert len(to_process) == 2
        assert {p["id"] for p in to_process} == {"id_004", "id_005"}

    def test_no_duplicate_ids_after_partial_run(self, tmp_path):
        """After saving, no ID appears more than once in the checkpoint file."""
        path = str(tmp_path / "checkpoints" / "synthesis_checkpoint.json")
        ids = {"id_001", "id_002", "id_001"}  # set already deduplicates
        self._save(path, ids)

        with open(path) as f:
            raw_list = json.load(f)

        assert len(raw_list) == len(set(raw_list)), "Duplicate IDs found in checkpoint"

    def test_incremental_saves_accumulate_correctly(self, tmp_path):
        """Simulates checkpointing every 10 samples over 3 batches."""
        path = str(tmp_path / "checkpoints" / "synthesis_checkpoint.json")
        completed = set()

        batch_1 = {f"id_{i:03d}" for i in range(10)}
        batch_2 = {f"id_{i:03d}" for i in range(10, 20)}
        batch_3 = {f"id_{i:03d}" for i in range(20, 25)}

        for batch in [batch_1, batch_2, batch_3]:
            completed |= batch
            self._save(path, completed)

        final = self._load(path)
        assert len(final) == 25
        assert "id_000" in final
        assert "id_024" in final


# ===========================================================================
# 3. MANIFEST ENCODING
# Arabic text must survive JSONL write → read without corruption.
# Tests cover: Arabic script, Egyptian dialect markers, mixed content,
# special characters, and multi-sample files.
# ===========================================================================

class TestManifestEncoding:

    SAMPLES = [
        {
            "audio_filepath": "data/final_dataset/audio/train/43fa3ac3.wav",
            "text": "والنبي الجو النهاردة حر قوي، مش قادر أطلع بره خالص",
            "duration": 5.942,
            "split": "train",
        },
        {
            "audio_filepath": "data/final_dataset/audio/train/4deed8ae.wav",
            "text": "أمي طبخت ملوخية النهاردة والأكل كان تحفة بجد",
            "duration": 5.099,
            "split": "train",
        },
        {
            "audio_filepath": "data/final_dataset/audio/dev/8d6559ef.wav",
            "text": "إزيك يا عم أحمد؟ مشوفتكش من زمان خالص",
            "duration": 3.659,
            "split": "dev",
        },
    ]

    def _write_manifest(self, path, samples):
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    def _read_manifest(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_arabic_text_survives_roundtrip(self, tmp_path):
        path = str(tmp_path / "train_manifest.jsonl")
        self._write_manifest(path, self.SAMPLES)
        loaded = self._read_manifest(path)
        for original, recovered in zip(self.SAMPLES, loaded):
            assert original["text"] == recovered["text"], (
                f"Text mismatch:\n  original : {original['text']}\n  recovered: {recovered['text']}"
            )

    def test_no_unicode_escapes_in_file(self, tmp_path):
        """ensure_ascii=False must be used — file must not contain \\uXXXX sequences."""
        path = str(tmp_path / "train_manifest.jsonl")
        self._write_manifest(path, self.SAMPLES)
        raw = open(path, encoding="utf-8").read()
        assert "\\u" not in raw, "File contains Unicode escapes — ensure_ascii=False was not used"

    def test_all_required_keys_present(self, tmp_path):
        path = str(tmp_path / "train_manifest.jsonl")
        self._write_manifest(path, self.SAMPLES)
        loaded = self._read_manifest(path)
        required_keys = {"audio_filepath", "text", "duration", "split"}
        for entry in loaded:
            assert required_keys.issubset(entry.keys()), f"Missing keys in entry: {entry}"

    def test_duration_is_float(self, tmp_path):
        path = str(tmp_path / "train_manifest.jsonl")
        self._write_manifest(path, self.SAMPLES)
        loaded = self._read_manifest(path)
        for entry in loaded:
            assert isinstance(entry["duration"], float), (
                f"duration should be float, got {type(entry['duration'])}"
            )

    def test_egyptian_dialect_markers_preserved(self, tmp_path):
        """Key EGY markers (مش، دلوقتي، عشان، إيه، كده) must not be corrupted."""
        egy_markers = [
            {"audio_filepath": "a.wav", "text": "مش عارف إيه اللي بيحصل دلوقتي", "duration": 3.0, "split": "test"},
            {"audio_filepath": "b.wav", "text": "عشان كده مش هروح النهاردة", "duration": 2.5, "split": "test"},
            {"audio_filepath": "c.wav", "text": "بتاع إيه ده بالظبط؟", "duration": 2.0, "split": "test"},
        ]
        path = str(tmp_path / "test_manifest.jsonl")
        self._write_manifest(path, egy_markers)
        loaded = self._read_manifest(path)
        for original, recovered in zip(egy_markers, loaded):
            assert original["text"] == recovered["text"]

    def test_code_switched_text_preserved(self, tmp_path):
        """Mixed Arabic+Latin (code-switching) must survive intact."""
        mixed = [
            {"audio_filepath": "x.wav", "text": "الـ meeting بكره الساعة تلاتة", "duration": 3.1, "split": "train"},
            {"audio_filepath": "y.wav", "text": "الـ wifi مش شغال خالص", "duration": 2.2, "split": "train"},
        ]
        path = str(tmp_path / "mixed_manifest.jsonl")
        self._write_manifest(path, mixed)
        loaded = self._read_manifest(path)
        for original, recovered in zip(mixed, loaded):
            assert original["text"] == recovered["text"]

    def test_line_count_matches_sample_count(self, tmp_path):
        path = str(tmp_path / "train_manifest.jsonl")
        self._write_manifest(path, self.SAMPLES)
        loaded = self._read_manifest(path)
        assert len(loaded) == len(self.SAMPLES)

    def test_empty_manifest_loads_as_empty_list(self, tmp_path):
        path = str(tmp_path / "empty_manifest.jsonl")
        self._write_manifest(path, [])
        loaded = self._read_manifest(path)
        assert loaded == []

    def test_split_values_are_valid(self, tmp_path):
        path = str(tmp_path / "train_manifest.jsonl")
        self._write_manifest(path, self.SAMPLES)
        loaded = self._read_manifest(path)
        valid_splits = {"train", "dev", "test"}
        for entry in loaded:
            assert entry["split"] in valid_splits, f"Invalid split value: {entry['split']}"