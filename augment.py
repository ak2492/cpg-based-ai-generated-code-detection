"""Adversarial training augmentation — exact per-notebook logic.

Python/C++ use regex obfuscation (apply_source_obfuscation_*); Java uses the
tree-sitter full-attack chain. The 20% (all) + 20% (AI-only) schedule,
tqdm descs and print wording are preserved verbatim per language.
"""
import random
from tqdm.auto import tqdm
from datasets import Dataset, concatenate_datasets

from attack_utils import (
    apply_source_obfuscation_python,
    apply_source_obfuscation_cpp,
    apply_full_attack_basic_java,
    apply_full_attack_enhanced_java,
)


def generate_adversarial_augmentations(train_clean_data, language, parser=None):
    code_col, label_col = 'code', 'label'
    aug_samples = []

    if language == "python":
        n_basic = int(0.20 * len(train_clean_data))
        aug_indices_basic = random.sample(range(len(train_clean_data)), min(n_basic, len(train_clean_data)))
        for idx in tqdm(aug_indices_basic, desc="Synthesizing Step 1 Adv Augmentations (All)"):
            row = train_clean_data[idx]
            aug_samples.append({code_col: apply_source_obfuscation_python(row[code_col]), label_col: row[label_col]})

        ai_indices = [i for i, row in enumerate(train_clean_data) if int(row[label_col]) == 1]
        n_enhanced = int(0.20 * len(train_clean_data))
        aug_indices_enhanced = random.sample(ai_indices, min(n_enhanced, len(ai_indices)))
        for idx in tqdm(aug_indices_enhanced, desc="Synthesizing Step 2 Adv Augmentations (AI Only)"):
            row = train_clean_data[idx]
            aug_samples.append({code_col: apply_source_obfuscation_python(row[code_col]), label_col: row[label_col]})

        train_adv_data = concatenate_datasets([train_clean_data, Dataset.from_list(aug_samples)])
        print(f"Clean Training Set: {len(train_clean_data)} | Augmented Training Set: {len(train_adv_data)}")
        return train_adv_data

    elif language == "cpp":
        n_basic = int(0.20 * len(train_clean_data))
        aug_indices_basic = random.sample(range(len(train_clean_data)), min(n_basic, len(train_clean_data)))
        for idx in tqdm(aug_indices_basic, desc="Synthesizing Step 1 Adv Augmentations (All)"):
            row = train_clean_data[idx]
            aug_samples.append({code_col: apply_source_obfuscation_cpp(row[code_col]), label_col: row[label_col]})

        ai_indices = [i for i, row in enumerate(train_clean_data) if int(row[label_col]) == 1]
        n_enhanced = int(0.20 * len(train_clean_data))
        aug_indices_enhanced = random.sample(ai_indices, min(n_enhanced, len(ai_indices)))
        for idx in tqdm(aug_indices_enhanced, desc="Synthesizing Step 2 Adv Augmentations (AI Only)"):
            row = train_clean_data[idx]
            aug_samples.append({code_col: apply_source_obfuscation_cpp(row[code_col]), label_col: row[label_col]})

        train_adv_data = concatenate_datasets([train_clean_data, Dataset.from_list(aug_samples)])
        print(f"Clean Training Set: {len(train_clean_data)} | Augmented Training Set: {len(train_adv_data)}")
        return train_adv_data

    else:  # java
        n_basic = int(0.20 * len(train_clean_data))
        aug_indices_basic = random.sample(range(len(train_clean_data)), min(n_basic, len(train_clean_data)))
        for idx in tqdm(aug_indices_basic, desc="Synthesizing Basic Adv Augmentations"):
            row = train_clean_data[idx]
            aug_samples.append({code_col: apply_full_attack_basic_java(row[code_col], parser), label_col: row[label_col]})

        ai_indices = [i for i, row in enumerate(train_clean_data) if int(row[label_col]) == 1]
        n_enhanced = int(0.20 * len(train_clean_data))
        aug_indices_enhanced = random.sample(ai_indices, min(n_enhanced, len(ai_indices)))
        for idx in tqdm(aug_indices_enhanced, desc="Synthesizing Enhanced Adv Augmentations"):
            row = train_clean_data[idx]
            aug_samples.append({code_col: apply_full_attack_enhanced_java(row[code_col], parser), label_col: row[label_col]})

        train_adv_data = concatenate_datasets([train_clean_data, Dataset.from_list(aug_samples)])
        print(f"Clean Training Pool: {len(train_clean_data)} | Augmented Training Pool: {len(train_adv_data)}")
        return train_adv_data
