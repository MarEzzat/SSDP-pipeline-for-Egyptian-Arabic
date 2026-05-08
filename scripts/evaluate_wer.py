import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import whisper
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Text normalisation (Arabic-aware)

def normalise_arabic(text: str) -> str:
    # Strip Arabic diacritics (harakat + shadda + tanwin variants)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    # Remove tatweel
    text = re.sub(r"\u0640", "", text)
    # Normalize alef variants → bare alef
    text = re.sub(r"[أإآ]", "ا", text)
    # Normalize teh marbuta → heh
    text = re.sub(r"ة", "ه", text)
    # Lowercase Latin
    text = text.lower()
    # Collapse whitespace & strip
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenise(text: str) -> List[str]:
    # Split normalised text into tokens for WER calculation
    return text.split()

# WER / CER calculation

def _edit_distance(ref: List, hyp: List) -> int:
    # Standard dynamic-programming edit distance.
    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def compute_wer(reference: str, hypothesis: str) -> float:
     #Word Error Rate: edit_distance(words) / len(reference_words).
    ref = tokenise(normalise_arabic(reference))
    hyp = tokenise(normalise_arabic(hypothesis))
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def compute_cer(reference: str, hypothesis: str) -> float:
    # Character Error Rate: edit_distance(chars) / len(reference_chars).
    ref = list(normalise_arabic(reference).replace(" ", ""))
    hyp = list(normalise_arabic(hypothesis).replace(" ", ""))
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return _edit_distance(ref, hyp) / len(ref)

# Data structures

@dataclass
class SampleResult:
    id: str
    audio_filepath: str
    reference: str
    hypothesis: str
    wer: float
    cer: float
    duration: float
    passed: bool          # WER < threshold
    whisper_language: str


@dataclass
class EvalReport:
    model: str
    manifest: str
    total_samples: int
    evaluated: int
    failed_transcription: int
    mean_wer: float
    median_wer: float
    mean_cer: float
    pass_rate: float       # % samples with WER < threshold
    wer_threshold: float
    total_audio_seconds: float
    rtf: float             # real-time factor (transcription time / audio time)
    samples: List[dict]


# Main evaluator
class WEREvaluator:

    def __init__(self, model_name: str = "large-v2", wer_threshold: float = 0.30):
        self.wer_threshold = wer_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Loading Whisper {model_name} on {self.device} ...")
        self.model = whisper.load_model(model_name, device=self.device)
        log.info("Model ready.")

    def transcribe(self, audio_path: str) -> tuple[str, str]:
        """
        Returns (hypothesis_text, detected_language).
        Forces Arabic decoding so Whisper doesn't defect to MSA romanisation.
        """
        result = self.model.transcribe(
            audio_path,
            language="ar",          # force Arabic
            task="transcribe",
            fp16=(self.device == "cuda"),
            temperature=0.0,        # greedy — deterministic output
        )
        return result["text"].strip(), result.get("language", "ar")

    def evaluate(
        self,
        manifest_path: str,
        limit: Optional[int] = None,
        output_path: str = "logs/wer_report.json",
    ) -> EvalReport:

        log.info(f"Reading manifest: {manifest_path}")
        samples = []
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        if limit:
            samples = samples[:limit]
            log.info(f"Limited to first {limit} samples.")

        log.info(f"Evaluating {len(samples)} samples ...")

        results: List[SampleResult] = []
        failed = 0
        wall_start = time.time()
        total_audio_seconds = 0.0

        for i, sample in enumerate(samples):
            audio_path = sample["audio_filepath"]
            reference = sample["text"]
            duration = float(sample.get("duration", 0.0))
            total_audio_seconds += duration

            # Derive a stable ID
            sample_id = os.path.splitext(os.path.basename(audio_path))[0]

            if not os.path.exists(audio_path):
                log.warning(f"[{i+1}/{len(samples)}] MISSING: {audio_path}")
                failed += 1
                continue

            try:
                hypothesis, lang = self.transcribe(audio_path)
            except Exception as e:
                log.error(f"[{i+1}/{len(samples)}] Transcription failed for {audio_path}: {e}")
                failed += 1
                continue

            wer = compute_wer(reference, hypothesis)
            cer = compute_cer(reference, hypothesis)
            passed = wer < self.wer_threshold

            result = SampleResult(
                id=sample_id,
                audio_filepath=audio_path,
                reference=reference,
                hypothesis=hypothesis,
                wer=round(wer, 4),
                cer=round(cer, 4),
                duration=duration,
                passed=passed,
                whisper_language=lang,
            )
            results.append(result)

            status = "OKK" if passed else "NO"
            log.info(
                f"[{i+1}/{len(samples)}] {status} WER={wer:.2%}  CER={cer:.2%}  "
                f"REF: {reference[:50]}..."
            )

        wall_time = time.time() - wall_start
        rtf = wall_time / total_audio_seconds if total_audio_seconds > 0 else 0.0

        # Aggregate stats
        wers = [r.wer for r in results]
        cers = [r.cer for r in results]

        import statistics
        mean_wer = statistics.mean(wers) if wers else 0.0
        median_wer = statistics.median(wers) if wers else 0.0
        mean_cer = statistics.mean(cers) if cers else 0.0
        pass_rate = sum(1 for r in results if r.passed) / len(results) if results else 0.0

        report = EvalReport(
            model=str(self.model),
            manifest=manifest_path,
            total_samples=len(samples),
            evaluated=len(results),
            failed_transcription=failed,
            mean_wer=round(mean_wer, 4),
            median_wer=round(median_wer, 4),
            mean_cer=round(mean_cer, 4),
            pass_rate=round(pass_rate, 4),
            wer_threshold=self.wer_threshold,
            total_audio_seconds=round(total_audio_seconds, 2),
            rtf=round(rtf, 4),
            samples=[asdict(r) for r in results],
        )

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)

        self._print_summary(report, output_path)
        return report

    @staticmethod
    def _print_summary(report: EvalReport, output_path: str):
        print("\n" + "=" * 60)
        print("  SSDP — WER EVALUATION REPORT")
        print("=" * 60)
        print(f"  Manifest        : {report.manifest}")
        print(f"  Samples         : {report.evaluated} evaluated / {report.total_samples} total")
        print(f"  Failed (no file): {report.failed_transcription}")
        print(f"  Audio duration  : {report.total_audio_seconds:.1f}s "
              f"({report.total_audio_seconds/60:.1f} min)")
        print(f"  RTF             : {report.rtf:.2f}x real-time")
        print("-" * 60)
        print(f"  Mean  WER       : {report.mean_wer:.2%}")
        print(f"  Median WER      : {report.median_wer:.2%}")
        print(f"  Mean  CER       : {report.mean_cer:.2%}")
        print(f"  Pass rate       : {report.pass_rate:.1%}  "
              f"(WER < {report.wer_threshold:.0%})")
        print("-" * 60)
        print(f"  Full report     : {output_path}")
        print("=" * 60)

        print()

# CLI entry point

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SSDP test split with Whisper WER proxy"
    )
    parser.add_argument(
        "--manifest",
        default="data/final_dataset/test_manifest.jsonl",
        help="Path to JSONL manifest (default: test split)",
    )
    parser.add_argument(
        "--model",
        default="large-v2",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size (default: large-v2)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="WER threshold for pass/fail (default: 0.30)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate first N samples (for quick smoke-tests)",
    )
    parser.add_argument(
        "--output",
        default="logs/wer_report.json",
        help="Output path for JSON report",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    evaluator = WEREvaluator(model_name=args.model, wer_threshold=args.threshold)
    evaluator.evaluate(
        manifest_path=args.manifest,
        limit=args.limit,
        output_path=args.output,
    )