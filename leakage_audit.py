"""SHA256 leakage & contamination audit — exact notebook logic per language.

normalize_*_for_hashing bodies are identical (whitespace strip); per-language
wrappers are preserved so messages match each notebook verbatim. Raw rows
reload cheaply (no graph building); run after main.py so the same raw pools
are hashed.

Usage:
  python leakage_audit.py --language python
  python leakage_audit.py --language cpp
"""
import argparse
import hashlib
import re

from pipeline import load_audit_raw_rows


def _normalize_for_hashing(code_str):
    return re.sub(r'\s+', '', str(code_str))


def normalize_python_for_hashing(code_str):
    return re.sub(r'\s+', '', str(code_str))


def normalize_cpp_for_hashing(code_str):
    return re.sub(r'\s+', '', str(code_str))


def normalize_java_for_hashing(code_str):
    return re.sub(r'\s+', '', str(code_str))


def compute_split_hashes(split_name, dataset_iterable, normalize_fn=None):
    normalize_fn = normalize_fn or _normalize_for_hashing
    hashes = set()
    for row in dataset_iterable:
        code = row.get('code') or row.get('text') or ''
        if code:
            norm = normalize_fn(code)
            hashes.add(hashlib.sha256(norm.encode('utf-8')).hexdigest())
    print(f"[{split_name:<25}] Unique Hashes: {len(hashes)}")
    return hashes


def run_audit(language="python", trial_samples=None, limit=None):
    label = {"python": "PYTHON", "cpp": "C++", "java": "JAVA"}[language]
    print("\n" + "=" * 70)
    print(f"{label} DATA LEAKAGE & CONTAMINATION AUDIT (SHA256)")
    print("=" * 70)

    bundle = load_audit_raw_rows(language, trial_samples=trial_samples, limit=limit)
    norm_fn = {"python": normalize_python_for_hashing,
               "cpp": normalize_cpp_for_hashing,
               "java": normalize_java_for_hashing}[language]
    clean_tag = {"python": "Python Clean Train", "cpp": "C++ Clean Train", "java": "Java Clean Train"}[language]
    adv_tag = {"python": "Python Augmented Train", "cpp": "C++ Augmented Train", "java": "Java Augmented Train"}[language]
    val_tag = {"python": "Python Validation", "cpp": "C++ Validation", "java": "Java Validation"}[language]
    test_tag = {"python": "Python Balanced Test", "cpp": "C++ Balanced Test", "java": "Java Test"}[language]

    train_clean_hashes = compute_split_hashes(clean_tag, bundle["train_clean_data"], norm_fn)
    train_adv_hashes = compute_split_hashes(adv_tag, bundle["train_adv_data"], norm_fn)
    val_hashes = compute_split_hashes(val_tag, bundle["val_eval_data"], norm_fn)
    test_hashes = compute_split_hashes(test_tag, bundle["test_raw"], norm_fn)

    print("\n" + "-" * 70)
    print(f"{label} INTERSECTION METRICS" if language != "python" else "PYTHON INTERSECTION METRICS")
    print("-" * 70)

    leak_clean_val = train_clean_hashes.intersection(val_hashes)
    leak_clean_test = train_clean_hashes.intersection(test_hashes)
    leak_adv_test = train_adv_hashes.intersection(test_hashes)

    print(f"Clean Train <-> Validation Overlap : {len(leak_clean_val)} samples")
    print(f"Clean Train <-> Test Overlap       : {len(leak_clean_test)} samples")
    print(f"Adv. Train  <-> Test Overlap       : {len(leak_adv_test)} samples")

    if len(test_hashes) > 0:
        contamination_pct = (len(leak_adv_test) / len(test_hashes)) * 100
        print(f"\nTest Set Contamination Rate: {contamination_pct:.4f}%")
        if contamination_pct < 0.5:
            if language == "python":
                print("[✓] PASSED: Contamination is statistically negligible (below 0.5%).")
            else:
                print("[✓] PASSED: Contamination is statistically negligible.")
        else:
            print("[!] WARNING: Elevated contamination detected across splits." if language == "python"
                  else "[!] WARNING: Contamination detected across splits.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="python", choices=["python", "java", "cpp"])
    parser.add_argument("--trial-samples", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_audit(args.language, args.trial_samples, args.limit)
