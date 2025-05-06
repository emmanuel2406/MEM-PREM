import json
import re
from datetime import datetime
import numpy as np
from collections import defaultdict

def parse_timestamp(ts):
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")

def extract_token_counts(record):
    try:
        raw_output = record["output"]
        
        # Extract prompt_token_ids using regex
        prompt_tokens_match = re.search(r'prompt_token_ids=\[(.*?)\]', raw_output)
        prompt_tokens = []
        if prompt_tokens_match:
            prompt_tokens_str = prompt_tokens_match.group(1)
            if prompt_tokens_str:
                prompt_tokens = [int(t) for t in prompt_tokens_str.split(', ')]
        
        # Extract token_ids from outputs
        token_ids_match = re.search(r'token_ids=\[(.*?)\]', raw_output)
        output_tokens = []
        if token_ids_match:
            token_ids_str = token_ids_match.group(1)
            if token_ids_str:
                output_tokens = [int(t) for t in token_ids_str.split(', ')]
        
        return len(prompt_tokens), len(output_tokens)
    except Exception as e:
        print(f"Error extracting token counts: {e}")
        return 0, 0

def preprocess_json(json_path):
    MAX_LEN = 800
    BINS = 12
    with open(json_path, "r") as f:
        data = json.load(f)

    records = data.get("records", [])
    print("Number of records=", len(records))
    token_rates = []
    total_lengths = []

    for r in records:
        if "output_ts" not in r or "output" not in r:
            continue

        try:
            start = parse_timestamp(r["start_ts"])
            end = parse_timestamp(r["output_ts"])
            elapsed = (end - start).total_seconds()
            if elapsed <= 0:
                continue

            prompt_len, output_len = extract_token_counts(r)
            total_tokens = prompt_len + output_len

            if total_tokens == 0:
                continue

            token_rates.append(total_tokens / elapsed)
            total_lengths.append(total_tokens)

        except Exception as e:
            continue

    # Display token generation stats
    print("Estimated Token Generation Rate (tokens/sec):")
    print(f"  Mean:   {np.mean(token_rates):.2f}")
    print(f"  Median: {np.median(token_rates):.2f}")
    print(f"  Std:    {np.std(token_rates):.2f}")
    print()

    # Create BINS equal-width bin edges between 0 and MAX_LEN as in the TRAIL paper
    log_min = 6
    log_max = np.log2(MAX_LEN)
    bin_edges = np.logspace(log_min, log_max, num=BINS + 1, base=2)  # BINS + 1 edges → BINS bins
    bin_labels = [f"{int(bin_edges[i])}–{int(bin_edges[i+1])}" for i in range(BINS)]
    bin_counts = dict.fromkeys(bin_labels, 0)

    # Bin the data
    for L in total_lengths:
        for i in range(BINS):
            if bin_edges[i] <= L < bin_edges[i+1]:
                bin_counts[bin_labels[i]] += 1
                break
            elif i == BINS-1 and L == MAX_LEN:  # Include right edge
                bin_counts[bin_labels[i]] += 1

    # Print the results
    print(f"Length Distribution ({BINS} bins between {log_min}–{MAX_LEN}):")
    total = sum(bin_counts.values())
    for label in bin_labels:
        count = bin_counts[label]
        print(f"  {label:<10}: {count} ({100 * count / total:.1f}%)")

if __name__ == "__main__":
    preprocess_json("llama3-4k.json")