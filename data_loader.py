import random


CODE_COL, LABEL_COL = 'code', 'label'


def balanced_subset(split, seed=42):
    labels = [int(row[LABEL_COL]) for row in split]
    idx0 = [i for i, y in enumerate(labels) if y == 0]
    idx1 = [i for i, y in enumerate(labels) if y == 1]
    rng = random.Random(seed)
    chosen = rng.sample(idx0, min(len(idx0), len(idx1))) + rng.sample(idx1, min(len(idx0), len(idx1)))
    rng.shuffle(chosen)
    return split.select(chosen)


def load_code_data(language="python", split="train", limit=None, trial_samples=None):
    """Mirror of notebook loading: HungPhamBKCS/magecode-dataset + optional TRIAL_SAMPLES + limit."""
    from datasets import load_dataset
    dataset = load_dataset("HungPhamBKCS/magecode-dataset", language)
    data = dataset[split]
    if trial_samples is not None:
        data = data.shuffle(seed=42).select(range(min(trial_samples, len(data))))
    if limit is not None:
        data = data.select(range(min(limit, len(data))))
    codes = [row.get('text', row.get('code', '')) for row in data]
    labels = [row.get('label', row.get('target', 0)) for row in data]
    return codes, labels


def load_magecode_splits(language="python", trial_samples=None):
    """Return (train_data, val_data, test_data) exactly as notebooks do."""
    from datasets import load_dataset
    dataset = load_dataset("HungPhamBKCS/magecode-dataset", language)
    train_data, val_data, test_data = dataset['train'], dataset['validation'], dataset['test']
    if trial_samples is not None:
        train_data = train_data.shuffle(seed=42).select(range(min(trial_samples, len(train_data))))
        val_data = val_data.shuffle(seed=42).select(range(min(trial_samples, len(val_data))))
        test_data = test_data.shuffle(seed=42).select(range(min(trial_samples, len(test_data))))
    return train_data, val_data, test_data
