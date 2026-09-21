"""Shared data pipeline: load -> balance -> augment -> vocab -> graphs -> normalize.

Mirrors Cell 1 of each notebook exactly (prints, descs, test-balancing rule).
`prepare_graphs()` is used by main.py / train.py / evaluate.py so the
notebook flow is preserved while exposing a hybrid-folder style CLI.
"""
import random

import numpy as np
import torch

from data_loader import balanced_subset, load_magecode_splits
from language_configs import SEED, get_parser
from graph_builder import (
    GraphContext,
    fit_vocab_and_tokenizer,
    process_split,
    fit_normalization,
    apply_normalization,
)
from augment import generate_adversarial_augmentations


def _seed_everything():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True


SPLIT_DESC = {
    "python": ("Parsing Clean Baseline Python Graphs", "Parsing Augmented Python Train Graphs",
               "Parsing Validation Python Graphs", "Parsing Rebalanced Python Test Graphs"),
    "cpp": ("Parsing Clean Baseline C++ Graphs", "Parsing Augmented C++ Train Graphs",
            "Parsing Validation C++ Graphs", "Parsing Rebalanced C++ Test Graphs"),
    "java": ("Parsing Clean Baseline Train Graphs", "Parsing Augmented Train Graphs",
             "Parsing Validation Graphs", "Parsing Clean Test Graphs"),
}

NORM_MSG = {
    "python": "[✓] Python Feature normalization locked using Clean Baseline statistics.",
    "cpp": "[✓] C++ Feature normalization locked using Clean Baseline statistics.",
    "java": "[✓] Feature normalization locked using Clean Baseline statistics.",
}

LOADED_MSG = {
    "python": "Loaded Python Splits: Train={t}, Val={v}, Test={te}",
    "java": "Loaded Java Splits: Train={t}, Val={v}, Test={te}",
    "cpp": "Loaded C++ Splits: Train={t}, Val={v}, Test={te}",
}


def prepare_graphs(language="python", trial_samples=None, limit=None):
    """Full Cell-1 pipeline. `limit` caps balanced splits for quick debugging (None = notebook exact)."""
    _seed_everything()
    print(f"Loading full {language.capitalize()} dataset from Hugging Face..."
          if language != "cpp" else "Loading full C++ dataset from Hugging Face...")

    train_data, val_data, test_data = load_magecode_splits(language, trial_samples=trial_samples)
    print(LOADED_MSG[language].format(t=len(train_data), v=len(val_data), te=len(test_data)))

    parser, _ = get_parser(language)

    train_clean_data = balanced_subset(train_data, SEED)
    val_eval_data = balanced_subset(val_data, SEED + 1)

    if limit is not None:
        train_clean_data = train_clean_data.select(range(min(limit, len(train_clean_data))))
        val_eval_data = val_eval_data.select(range(min(limit, len(val_eval_data))))

    train_adv_data = generate_adversarial_augmentations(train_clean_data, language, parser)

    type_to_id, vocab_size, bpe_tokenizer, pad_id, token_to_rank, max_rank = fit_vocab_and_tokenizer(
        train_clean_data, parser, language)
    ctx = GraphContext(language, parser, type_to_id, vocab_size, bpe_tokenizer, pad_id, token_to_rank, max_rank)

    d_clean, d_adv, d_val, d_test = SPLIT_DESC[language]
    train_clean_graphs = process_split(train_clean_data, d_clean, ctx)
    train_adv_graphs = process_split(train_adv_data, d_adv, ctx)
    val_graphs = process_split(val_eval_data, d_val, ctx)

    if language in ("python", "cpp"):
        # Balanced Test set extraction to eliminate base-rate skew
        test_balanced_raw = balanced_subset(test_data, seed=42)
        if limit is not None:
            test_balanced_raw = test_balanced_raw.select(range(min(limit, len(test_balanced_raw))))
        test_graphs = process_split(test_balanced_raw, d_test, ctx)
    else:
        test_balanced_raw = test_data
        if limit is not None:
            test_balanced_raw = test_balanced_raw.select(range(min(limit, len(test_balanced_raw))))
        test_graphs = process_split(test_balanced_raw, d_test, ctx)

    fit_normalization(train_clean_graphs, ctx)
    for split in (train_clean_graphs, train_adv_graphs, val_graphs, test_graphs):
        apply_normalization(split, ctx)

    print(NORM_MSG[language])
    return {
        "ctx": ctx,
        "parser": parser,
        "train_clean_data": train_clean_data,
        "train_adv_data": train_adv_data,
        "val_eval_data": val_eval_data,
        "test_raw": test_balanced_raw,
        "test_full": test_data,
        "train_clean_graphs": train_clean_graphs,
        "train_adv_graphs": train_adv_graphs,
        "val_graphs": val_graphs,
        "test_graphs": test_graphs,
    }
