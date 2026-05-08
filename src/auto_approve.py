import json
import os

def auto_approve_samples():
    print("=" * 70)
    print("=" * 70)
    print()
    
    # Load synthesis results
    samples = []
    with open('data/synthesis_results.jsonl', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f" Loaded {len(samples)} synthesized samples")
    
    # Load existing reviews (if any)
    try:
        with open('data/reviews.json', encoding='utf-8') as f:
            reviews = json.load(f)
        print(f" Loaded {len(reviews)} existing reviews")
    except:
        reviews = {}
        print(" No existing reviews found, starting fresh")
    
    print()
    print(" Processing samples...")
    print()
    
    # Auto-approve logic
    auto_approved = 0
    flagged = []
    already_reviewed = 0
    
    for s in samples:
        # Skip if already reviewed
        if s['id'] in reviews:
            already_reviewed += 1
            continue
        
        # Auto-approve clean samples
        if not s['flags']:
            reviews[s['id']] = {
                'decision': '✅ Approve',
                'notes': 'auto-approved (no quality flags)',
                'audio_path': s['audio_path'],
                'text': s['text'],
                'duration': s['duration']
            }
            auto_approved += 1
        else:
            # Flag for manual review
            flagged.append(s)
    
    # Save reviews
    with open('data/reviews.json', 'w', encoding='utf-8') as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)
    
    # Save flagged samples for easy manual review
    if flagged:
        with open('data/flagged_for_manual_review.jsonl', 'w', encoding='utf-8') as f:
            for s in flagged:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
    
    # Generate report
    print(" Auto-Approval Results:")
    print(f"   Auto-approved: {auto_approved}")
    print(f"   Flagged for manual review: {len(flagged)}")
    print(f"   Already reviewed: {already_reviewed}")
    print()
    
    if flagged:
        print(" Flagged Samples (need manual review):")
        print("-" * 70)
        for s in flagged[:10]:  # Show first 10
            flags_str = ', '.join(s['flags'])
            text_preview = s['text'][:60] + '...' if len(s['text']) > 60 else s['text']
            print(f"   [{s['id']}] {flags_str}")
            print(f"      Text: {text_preview}")
            print(f"      Duration: {s['duration']:.2f}s")
            print()
        
        if len(flagged) > 10:
            print(f"   ... and {len(flagged) - 10} more")
            print()
        
        print(f"  Tip: Run 'streamlit run src/3_review_ui.py' to manually review these")
        print(f"   Or simply reject all flagged: python src/reject_flagged.py")
        print()
    
    # Final counts
    approved = [k for k, v in reviews.items() if 'Approve' in v['decision']]
    rejected = [k for k, v in reviews.items() if 'Reject' in v['decision']]
    flagged_count = [k for k, v in reviews.items() if 'Flag' in v['decision']]
    
    print("=" * 70)
    print(" FINAL REVIEW STATISTICS:")
    print("=" * 70)
    print(f"    Approved: {len(approved)} ({len(approved)/len(samples)*100:.1f}%)")
    print(f"    Rejected: {len(rejected)} ({len(rejected)/len(samples)*100:.1f}%)")
    print(f"    Flagged: {len(flagged_count)} ({len(flagged_count)/len(samples)*100:.1f}%)")
    print(f"    Total: {len(reviews)}/{len(samples)} reviewed")
    print()
    
    # Quality assessment
    approval_rate = len(approved) / len(samples) * 100
    if approval_rate >= 80:
        print(" Excellent! >80% approval rate - high quality synthesis")
    elif approval_rate >= 60:
        print(" Good. 60-80% approval - acceptable quality")
    else:
        print(" Warning: <60% approval - may need to review TTS settings")
    
    print()
    print(" Reviews saved to: data/reviews.json")
    if flagged:
        print(" Flagged samples saved to: data/flagged_for_manual_review.jsonl")
    print()
    print("=" * 70)

if __name__ == "__main__":
    auto_approve_samples()