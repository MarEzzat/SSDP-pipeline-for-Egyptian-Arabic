# Synthetic Speech Data Pipeline (SSDP) for Egyptian Arabic

---

## Executive Summary

This pipeline produces **training-ready synthetic Egyptian Arabic speech data** for STT model fine-tuning. It addresses the core challenge: **Egyptian Arabic (EGY) is critically underrepresented in off-the-shelf speech systems**, which are predominantly trained on Modern Standard Arabic (MSA).

### Key Achievements

 **End-to-end pipeline**: From text generation → audio synthesis → quality review → NeMo-ready manifests  
 **Dialect-aware**: Prompts contain authentic EGY markers (مش، النهاردة، إزيك, عشان كده)  
 **Automated quality signals**: Too short/long audio, long text flags feed into review  
 **Checkpointing**: Long-running synthesis can resume from crashes  
 **WER validation**: 16% WER on Whisper large-v2 proves acoustic decodability  
 **Test coverage**: Critical logic (quality flags, checkpointing, Arabic encoding) is unit-tested  
 **Production-ready output**: JSONL manifests with proper UTF-8 encoding, ready for NeMo/Coqui

---

## Stage 1: Prompt Generation

### Approach

**Manual + Automated Hybrid**

 **GPT-4 Generation** (manual step): I crafted a detailed prompt asking GPT-4 to generate 500 Egyptian Arabic sentences across diverse domains, with explicit instructions to use EGY markers.

### Prompt Diversity Statistics

From generated corpus (500 prompts):

| Domain             | Count | % Total |
|--------------------|-------|---------|
| daily_conversation | 120   | 24%     |
| phone_calls        | 80    | 16%     |
| directions         | 70    | 14%     |
| food               | 65    | 13%     |
| complaints         | 55    | 11%     |
| shopping           | 50    | 10%     |
| work               | 40    | 8%      |
| instructions       | 20    | 4%      |

**Length distribution**:
- Short (1-5 words): 120 prompts
- Medium (6-15 words): 280 prompts
- Long (16-30 words): 100 prompts

**Complexity**:
- Simple sentences: 350
- Complex (subordinate clauses, conditionals): 150

### Egyptian Arabic Markers Verified

 **Negation**: مش (not ما)  
 **Present continuous**: بـ prefix (e.g., بتعمل إيه)  
 **Time expressions**: النهاردة، دلوقتي، بكره  
 **Question words**: إزيك، إيه، فين  
 **Connectives**: عشان (not لأن)  
 **Code-switching**: الـ wifi, الـ meeting (authentic EGY behavior)

### Sample Prompts

```json
{"id": "43fa3ac3", "text": "والنبي الجو النهاردة حر قوي، مش قادر أطلع بره خالص", "domain": "daily_conversation"}
{"id": "4deed8ae", "text": "أمي طبخت ملوخية النهاردة والأكل كان تحفة بجد", "domain": "food"}
{"id": "8d6559ef", "text": "إزيك يا عم أحمد؟ مشوفتكش من زمان خالص", "domain": "daily_conversation"}
{"id": "b2f1c8a9", "text": "ممكن تقولي فين أقرب صيدلية من هنا؟", "domain": "directions"}
{"id": "7a3d2e1f", "text": "الـ wifi مش شغال خالص، ممكن تشوف المشكلة؟", "domain": "complaints"}
```

---

## Stage 2: TTS Synthesis

### Model Selection: Coqui XTTSv2

**Why XTTSv2?**

| Criterion | XTTSv2 | Alternatives (Tacotron2, FastSpeech2) |
|-----------|--------|--------------------------------------|
| Arabic support |  Multilingual (24 langs incl. Arabic) |  MSA-only models |
| Zero-shot cloning |  Yes (reference voice) |  Requires speaker-specific fine-tuning |
| Prosody quality |  Natural (attention + flow-based) |  Robotic (autoregressive) |
| Inference speed |  Slow (~5s for 10s audio) |  Fast (real-time) |
| License |  Open source (Mozilla MPL 2.0) |  Open source |

**Trade-off accepted**: Slow synthesis (5 hours for 500 prompts) in exchange for naturalness and dialect adaptability.

### Egyptian Reference Voice

**Source**: Mozilla Common Voice 18 Arabic  
**Selection criteria**:
1. Accent field contains "Egypt" or "مصر"
2. Duration 3-15 seconds (enough context, not too long)
3. Good audio quality (RMS > 0.01, not silent)
4. Preferably male voice (better prosody transfer in XTTSv2)

**Extraction script**: `scripts/extract_egyptian_voice.py`
- Downloads Common Voice validation split (3GB, cached)
- Filters for Egyptian samples
- Resamples to 16kHz mono
- Saves to `assets/egyptian_speaker.wav`

**Fallback**: If no Egyptian voice found, use generic Arabic sample (documented in script).


### Automated Quality Flags

| Flag | Trigger | Rationale |
|------|---------|-----------|
| `too_short` | duration < 1.0s | Likely truncated audio; STT models struggle with <1s |
| `too_long` | duration > 20.0s | Memory-intensive for training; rare in real conversations |
| `long_text` | len(text) > 200 chars | Prosody degrades; higher risk of TTS artifacts |

These flags feed into the review UI, allowing reviewers to **prioritize flagged samples** for manual listening.

### Checkpointing & Resumability

**Problem**: Synthesis takes 5+ hours. A crash at sample 450 shouldn't force re-synthesis from scratch.

**Solution**: 
- Save `synthesis_checkpoint.json` every 10 samples
- On restart, load checkpoint and skip completed IDs
- Duplicate-safe: Use set for O(1) membership testing

**Test coverage**: `tests/test_pipeline.py::TestCheckpointIntegrity` verifies:
- Missing checkpoint returns empty set
- Save → load roundtrip preserves data
- Resuming skips already-completed IDs
- No duplicate IDs after partial runs

---

## Stage 3: Quality Review

### Approach: Streamlit UI + Auto-Approval

**Hybrid review strategy**:
1. **Auto-approve** samples with NO flags (clean audio, reasonable length)
2. **Manual review** for flagged samples (too_short, too_long, long_text)

**Why not 100% manual review?**
- 500 samples × 10s listening = 1.5 hours of pure playback time
- Clean samples (70-80%) don't need human attention
- Reviewers can focus on edge cases


**Key features**:
- **Progress tracking**: Visual progress bar + count
- **Contextual info**: Text, duration, domain, flags displayed together
- **Audio playback**: In-browser WAV player
- **Session persistence**: Reviews saved to `data/reviews.json` after each decision
- **Skip previously reviewed**: Automatically jumps to next unreviewed sample


### Review Decision Criteria

**Approve** :
- Audio is clear, no artifacts (clipping, noise, silence)
- Prosody sounds natural (intonation, pauses)
- Text-audio alignment is correct (no mismatches)

**Reject** :
- Severe artifacts (metallic voice, robotic prosody)
- Incorrect text (TTS hallucinated words)
- Audio file corrupted or silent

**Flag for manual check** :
- Minor prosody issues (slightly unnatural stress)
- Borderline duration (0.9s or 20.5s)
- Ambiguous quality (needs second opinion)

---

## Stage 4: Dataset Export

### Output Format: NeMo-Compatible JSONL

**Why JSONL?**
- **Standard format** for NeMo, Coqui STT, ESPnet, Whisper fine-tuning
- **Line-oriented**: Easy to stream, concatenate, shuffle
- **UTF-8 safe**: `ensure_ascii=False` preserves Arabic characters

**Manifest schema**:
```json
{
  "audio_filepath": "data/final_dataset/audio/train/43fa3ac3.wav",
  "text": "والنبي الجو النهاردة حر قوي",
  "duration": 5.942,
  "split": "train"
}
```


### Split Strategy

**80/10/10 stratification**:
- **Train (80%)**: Bulk of data for fine-tuning
- **Dev (10%)**: Validation during training (early stopping, hyperparameter tuning)
- **Test (10%)**: Final evaluation (WER reporting)

**No stratification by domain**: 
- Random shuffle is sufficient for 500 samples
- Domain imbalance (24% daily_conversation vs 4% instructions) would lead to tiny test sets for rare domains
- Real-world STT usage is domain-agnostic anyway

**Reproducibility**: `random.seed(42)` ensures splits are identical across runs.

### UTF-8 Encoding Verification

**Critical requirement**: Arabic text must survive JSONL write → read without corruption.

**Test coverage**: `tests/test_pipeline.py::TestManifestEncoding`
- Verifies `ensure_ascii=False` is used (no `\uXXXX` escapes in file)
- Confirms roundtrip: write → read → text matches
- Checks Egyptian dialect markers (مش، دلوقتي، عشان) are preserved
- Tests code-switched text (الـ wifi, الـ meeting)

---

## Quality Assurance & Validation

### 1. WER Evaluation (Proxy Metric)

**Why WER for a TTS pipeline?**
- **Acoustic decodability**: If Whisper can't transcribe it, an STT model can't learn from it
- **Quantitative signal**: Not just "it sounds okay" — we have a number
- **Industry standard**: WER is the universal STT metric


**Results**:
```
=== WER EVALUATION REPORT ===
Manifest        : data/final_dataset/test_manifest.jsonl
Samples         : 50 evaluated / 50 total
Mean WER        : 16.0%
Median WER      : 14.5%
Pass rate       : 94%  (WER < 30%)
```

**Interpretation**:
- **16% WER is excellent** for synthetic dialect data
- Comparable to real human recordings (10-20% WER on low-resource dialects)
- Proves audio is clean and acoustically decodable

**Arabic-aware normalization**:
- Removes diacritics (Whisper doesn't produce them)
- Normalizes alef variants (أ إ آ → ا)
- Normalizes teh marbuta (ة → ه)
- Lowercases Latin letters (for code-switching)

### 2. Unit Tests

**Test coverage**: `tests/test_pipeline.py` (pytest)

**What's tested**:

#### Quality Flag Logic (`TestQualityFlags`)
```python
def test_too_short_fires_below_1s():
    assert "too_short" in _get_flags(0.9, "نص قصير")

def test_too_long_fires_above_20s():
    assert "too_long" in _get_flags(20.1, "نص")

def test_long_text_fires_above_200_chars():
    assert "long_text" in _get_flags(5.0, "ا" * 201)
```

#### Checkpoint Integrity (`TestCheckpointIntegrity`)
```python
def test_missing_checkpoint_returns_empty_set():
    assert load_checkpoint("nonexistent.json") == set()

def test_save_then_load_roundtrip():
    save_checkpoint(path, {"id_1", "id_2"})
    assert load_checkpoint(path) == {"id_1", "id_2"}

def test_completed_ids_are_skipped_on_resume():
    # Simulate crash after 3/5 samples
    save_checkpoint(path, {"id_1", "id_2", "id_3"})
    completed = load_checkpoint(path)
    to_process = [p for p in all_prompts if p['id'] not in completed]
    assert to_process == [prompt_4, prompt_5]
```

#### Manifest Encoding (`TestManifestEncoding`)
```python
def test_arabic_text_survives_roundtrip():
    write_manifest(path, samples)
    loaded = read_manifest(path)
    assert loaded[0]['text'] == samples[0]['text']

def test_no_unicode_escapes_in_file():
    write_manifest(path, samples)
    raw = open(path).read()
    assert "\\u" not in raw  # Must use ensure_ascii=False

def test_egyptian_dialect_markers_preserved():
    samples = [{"text": "مش عارف إيه اللي بيحصل دلوقتي", ...}]
    write_manifest(path, samples)
    loaded = read_manifest(path)
    assert loaded[0]['text'] == samples[0]['text']
```

**Run tests**: `pytest tests/test_pipeline.py -v`

### 3. Intermediate Artifact Inspection

**Observable outputs at each stage**:
- `data/prompts.jsonl` → Verify prompt diversity, domain distribution
- `data/synthesis_results.jsonl` → Check flag rates, duration stats
- `data/reviews.json` → Approval/rejection rates
- `data/final_dataset/*_manifest.jsonl` → Final split sizes, encoding

---

## Egyptian Arabic Considerations

### Challenge 1: Dialect-Specific Lexicon

**Problem**: MSA-trained systems fail on EGY-specific words.

| MSA (Standard Arabic) | EGY (Egyptian) | English |
|-----------------------|----------------|---------|
| ليس | مش | not |
| الآن | دلوقتي | now |
| غدًا | بكره | tomorrow |
| لأن | عشان | because |
| كيف حالك؟ | إزيك؟ | How are you? |
| ماذا | إيه | what |

**Our solution**: 
- Explicit GPT-4 prompt: "Use Egyptian colloquial, not MSA"
- Post-generation validation: Grep for MSA-only particles (ليس، إن، لكن) → flag for review

### Challenge 2: Phonological Differences

**EGY != MSA pronunciation**:
- **Qaf (ق)** → glottal stop (ء) in EGY: قال → /ʾāl/ not /qāl/
- **Jīm (ج)** → /g/ in Cairo: جميل → /gamīl/ not /ǧamīl/
- **Vowel reduction**: ذهبت → /ru:ħt/ not /ðahabtu/

**TTS limitation**: XTTSv2 learns phonology from reference voice. If reference voice is MSA-speaker reading EGY text, prosody will be off.

**Mitigation**: 
- Extract true Egyptian speaker from Common Voice
- Fallback: Document that prosody may be imperfect if Egyptian voice unavailable

### Challenge 3: Code-Switching

**Real-world EGY behavior**: Mixing English tech terms into Arabic sentences.
- "الـ wifi مش شغال" (The wifi isn't working)
- "عندي meeting الساعة 3" (I have a meeting at 3)

**Pipeline handling**:
- GPT-4 prompted to include code-switching examples
- WER evaluation lowercases Latin chars to handle case mismatches

### Challenge 4: Orthography Variance

**EGY has no standardized spelling**:
- "بيعمل إيه" vs "بيعمل ايه" (both valid)
- Teh marbuta: "مدرسة" vs "مدرسه"

**Impact on STT training**: Orthographic inconsistency can confuse LM component.

**Our approach**: 
- Use GPT-4's consistent orthography (trained on large corpus)
- Normalize during WER eval (ة → ه)

---

## How to Run

### Prerequisites

```bash
# Python 3.9+
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate
 
# Install dependencies in correct order
pip install "numpy==1.22.4"
pip install "pyarrow==12.0.1"
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install TTS==0.22.0
pip install streamlit==1.28.0
pip install "librosa==0.10.1"
pip install soundfile==0.12.1
pip install "datasets==2.21.0"
pip install "transformers==4.33.0"
pip install pyyaml tqdm pandas
 
# Patch torch.load in TTS (PyTorch 2.6 breaking change)
(Get-Content "venv\lib\site-packages\TTS\utils\io.py") `
  -replace 'torch.load\(f, map_location=map_location', `
           'torch.load(f, map_location=map_location, weights_only=False' `
  | Set-Content "venv\lib\site-packages\TTS\utils\io.py"

```


### Step-by-Step Execution

#### Step 0: Extract Egyptian Reference Voice

```bash
python scripts/extract_egyptian_voice.py
```

**Output**: `assets/egyptian_speaker.wav` (16kHz mono)

#### Step 1: Generate Prompts

**Manual step**: Ask GPT-4 to generate 500 Egyptian Arabic prompts. Save JSON to `data/gpt4_prompts.json`.

Then process:
```bash
python src/1_generate_prompts.py
```

**Output**: `data/prompts.jsonl`

#### Step 2: Synthesize Audio

```bash
python src/2_synthesize_audio.py
```

**Output**: 
- `data/audio_raw/*.wav` (500 files)
- `data/synthesis_results.jsonl` (with quality flags)


#### Step 3: Review Samples

```bash
streamlit run src/3_review_ui.py
```

Opens browser at `http://localhost:8501`.

**Workflow**:
1. Listen to audio
2. Choose: Approve / Reject / Flag
3. Submit → Next sample
4. Progress saved after each decision

**Auto-approve shortcut** (before manual review):
```bash
python src/auto_approve.py
```

#### Step 4: Export Dataset

```bash
python src/4_export_dataset.py
```


#### Step 5: Validate with WER 

```bash
python scripts/evaluate_wer.py \
    --manifest data/final_dataset/test_manifest.jsonl \
    --model large-v2
```
---

## Observed Quality Issues & Limitations

### 1. Reference Voice Quality

**Issue**: Common Voice Egyptian samples are often low-quality (background noise, mic distortion).

**Impact**: TTS clones these artifacts → noisy synthetic audio.

**Mitigation**:
- Filter by RMS > 0.01 (not silent)
- Select longest sample with good volume
- Future: Manually record 10s Egyptian reference in studio

### 2. Prosody Degradation on Long Sentences

**Issue**: XTTSv2 prosody becomes robotic for >200 char sentences.

**Evidence**: `long_text` flag fires on 8% of samples; manual review confirms unnatural intonation.

**Mitigation**:
- Flag for review (not auto-approved)
- Consider splitting long prompts into clauses

### 3. Code-Switching Pronunciation

**Issue**: XTTSv2 sometimes pronounces English words with Arabic phonology ("wifi" → /wifiː/ with Arabic vowels).

**Impact**: Sounds unnatural to native speakers.

**No fix**: This is a model limitation. Real EGY speakers also vary in English pronunciation.

### 4. Homophone Confusion

**Example**: "مش" (not) vs "مشي" (walk) — TTS may stress incorrectly without context.

**Impact**: Rare but possible misalignment.

**Future work**: Add stress markers or contextual hints to prompts.

### 5. Domain Imbalance

**Issue**: 24% daily_conversation vs 4% instructions → test set may not cover rare domains.

**Impact**: STT model may underperform on instructions.

**Accepted trade-off**: Real-world usage is also biased toward casual conversation.

---

## Deliverable: Sample Output

**Location**: `sample_output/`

**Contents**:
- 50 samples from test split
- `sample_manifest.jsonl` with schema:
  ```json
  {
    "audio_filepath": "audio/43fa3ac3.wav",
    "text": "والنبي الجو النهاردة حر قوي",
    "duration": 5.942,
    "split": "test"
  }
  ```
- `audio/` folder with 50 WAV files (16kHz, mono)

**Usage**:
```bash
# Inspect manifest
head sample_output/sample_manifest.jsonl | jq .

# Play sample
ffplay sample_output/audio/43fa3ac3.wav

```

---

## Conclusion

This pipeline demonstrates:

1. **Sound engineering judgment**: Checkpointing, resumability, automated quality flags
2. **Dialect awareness**: Egyptian markers, code-switching, orthography challenges documented
3. **Quality focus**: WER validation (16%), test coverage, review workflow
4. **Production-ready output**: NeMo-compatible JSONL with proper UTF-8 encoding
5. **Transparency**: Limitations and trade-offs clearly documented

**Key insight**: The bottleneck isn't the model — it's the **data**. This pipeline prioritizes **quality** (dialect authenticity, acoustic decodability) over **quantity** (500 prompts is sufficient for fine-tuning validation).

**Next steps** (if productionized):
1. Scale to 10K prompts (GPT-4 → fine-tuned LLM for automation)
2. Record professional Egyptian reference voice (studio quality)
3. Add speaker diversity (male/female/age variants)
4. Implement active learning (fine-tune STT → identify hard samples → regenerate)

---

## Last But Not Least

**Mariam Ezzat**
**Email**: [mariamezzat578@gmail.com]
**GITHUB**: [github.com/MarEzzat]
