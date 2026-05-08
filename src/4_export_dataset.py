import json
import os
import random
import shutil
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def auto_approve_unflagged():
    """Auto-approve all non-flagged samples if reviews.json is empty or missing."""
    samples = []
    with open("data/synthesis_results.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    try:
        with open("data/reviews.json", encoding="utf-8") as f:
            reviews = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        reviews = {}

    auto_approved = 0
    for s in samples:
        if s["id"] not in reviews:
            if not s["flags"]:
                reviews[s["id"]] = {"decision": "Approve", "notes": "auto-approved"}
                auto_approved += 1

    with open("data/reviews.json", "w", encoding="utf-8") as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)

    if auto_approved > 0:
        logging.info(f"Auto-approved {auto_approved} unflagged samples")

    return reviews


def export_dataset():
    # Load config
    with open("config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Auto-approve unflagged samples if needed
    reviews = auto_approve_unflagged()

    # Load synthesis results
    samples = {}
    with open("data/synthesis_results.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                samples[s["id"]] = s

    # Get approved samples
    approved = []
    for sid, review in reviews.items():
        if "Approve" in review["decision"] and sid in samples:
            approved.append(samples[sid])

    logging.info(f"Approved samples: {len(approved)}")

    if not approved:
        logging.error("No approved samples found. Run review first.")
        return None

    # Shuffle and split
    random.seed(42)
    random.shuffle(approved)

    train_ratio = config.get("export", {}).get("splits", {}).get("train", 0.8)
    dev_ratio   = config.get("export", {}).get("splits", {}).get("dev",   0.1)

    train_size = int(len(approved) * train_ratio)
    dev_size   = int(len(approved) * dev_ratio)

    splits = {
        "train": approved[:train_size],
        "dev":   approved[train_size:train_size + dev_size],
        "test":  approved[train_size + dev_size:]
    }

    # Create output dirs
    for split in splits:
        os.makedirs(f"data/final_dataset/audio/{split}", exist_ok=True)

    # Export manifests and copy audio
    for split_name, split_samples in splits.items():
        manifest_path = f"data/final_dataset/{split_name}_manifest.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for s in split_samples:
                dst = f"data/final_dataset/audio/{split_name}/{s['id']}.wav"
                if os.path.exists(s["audio_path"]) and not os.path.exists(dst):
                    shutil.copy2(s["audio_path"], dst)
                entry = {
                    "audio_filepath": dst,
                    "text": s["text"],
                    "duration": s["duration"],
                    "split": split_name
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logging.info(f"  {split_name}: {len(split_samples)} samples → {manifest_path}")

    # Stats
    stats = {split: len(s) for split, s in splits.items()}
    stats["total"] = len(approved)
    with open("data/final_dataset/dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logging.info("Done! Dataset exported to data/final_dataset/")
    return stats


if __name__ == "__main__":
    stats = export_dataset()
    if stats:
        print("\n=== Export Complete ===")
        print(f"  Train : {stats['train']}")
        print(f"  Dev   : {stats['dev']}")
        print(f"  Test  : {stats['test']}")
        print(f"  Total : {stats['total']}")