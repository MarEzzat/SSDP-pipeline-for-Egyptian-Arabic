import json
import os
import yaml
from typing import List, Dict
import hashlib

def load_config():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def generate_id(text: str) -> str:
    """Generate unique ID from text hash"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:8]

def process_prompts(raw_prompts: List[Dict], config: Dict) -> List[Dict]:
    """
    Convert GPT-4 output to pipeline format
    
    Input (from GPT-4):
    [
        {
            "text": "ممكن تقولي فين أقرب صيدلية؟",
            "domain": "daily_conversation",
            "length_category": "short",
            "complexity": "simple"
        }
    ]
    
    Output (for pipeline):
    [
        {
            "id": "a3f2b9c1",
            "text": "ممكن تقولي فين أقرب صيدلية؟",
            "domain": "daily_conversation",
            "length_category": "short",
            "complexity": "simple",
            "char_count": 28,
            "word_count": 5
        }
    ]
    """
    processed = []
    
    for prompt in raw_prompts:
        text = prompt['text'].strip()
        
        # Generate unique ID
        prompt_id = generate_id(text)
        
        # Add metadata
        processed_prompt = {
            "id": prompt_id,
            "text": text,
            "domain": prompt.get('domain', 'unknown'),
            "length_category": prompt.get('length_category', 'medium'),
            "complexity": prompt.get('complexity', 'simple'),
            "char_count": len(text),
            "word_count": len(text.split())
        }
        
        processed.append(processed_prompt)
    
    return processed

def save_jsonl(data: List[Dict], output_path: str):
    """Save as JSONL (one JSON object per line)"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    print("=" * 60)
    
    # Load config
    config = load_config()
    input_file = config['prompts']['input_file']
    output_file = config['prompts']['output_file']
    
    # Check if GPT-4 output exists
    if not os.path.exists(input_file):
        print(f"\n ERROR: {input_file} not found!")
        print("\n MANUAL STEPS REQUIRED:")
        print("1. Copy this prompt to ChatGPT:")
        print("-" * 60)
        print(open('prompts/gpt4_prompt.txt', 'r').read())  
        print("-" * 60)
        print(f"\n2. Save ChatGPT's JSON output to: {input_file}")
        print(f"3. Run this script again\n")
        return
    
    # Load raw prompts from GPT-4
    print(f"\n Loading prompts from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_prompts = json.load(f)
    
    print(f" Found {len(raw_prompts)} prompts")
    
    # Process prompts
    print("\n** Processing prompts...")
    processed = process_prompts(raw_prompts, config)
    
    # Save as JSONL
    save_jsonl(processed, output_file)
    print(f" Saved to: {output_file}")
    
    # Statistics
    print("\n** Statistics:")
    print(f"   Total prompts: {len(processed)}")
    
    domains = {}
    for p in processed:
        domains[p['domain']] = domains.get(p['domain'], 0) + 1
    
    print(f"   Domains:")
    for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
        print(f"      - {domain}: {count}")
    
    avg_chars = sum(p['char_count'] for p in processed) / len(processed)
    avg_words = sum(p['word_count'] for p in processed) / len(processed)
    print(f"   Avg chars/prompt: {avg_chars:.1f}")
    print(f"   Avg words/prompt: {avg_words:.1f}")

if __name__ == "__main__":
    main()