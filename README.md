# CPG-Based AI-Generated Code Detection

A Code Property Graph (CPG) pipeline that mirrors the three desktop notebooks
(`astgnnpythondesktop.ipynb`, `astgnnjavadesktop.ipynb`, `astgnncppdesktop.ipynb`)
as an executable GitHub-style module, following the same CLI conventions as
`AI_Generated_Hybrid_Code_Detection/` (`main.py` / `train.py` / `evaluate.py` /
`attack_*.py`).

Each language builds 34-d node stylometry + 16-d macro features + BPE subwords
into a 16-relation graph (AST parent/child, siblings, use-chains, leaf order,
virtual node, def-use, CFG, call, arg-param) encoded by a 4-layer gated RGCN
(`AdvancedASTGraphEncoder`) with FiLM global conditioning.

---

## Installation

Step 1 — everything except `torch-scatter` (fast, no compilation):

```bash
pip install -r requirements.txt
```

Step 2 — `torch-scatter` from the PyG wheel index, exactly like notebook
Cell 0 (required by `RGCNConv` in `model.py`; do NOT `pip install torch-scatter`
bare — PyPI only has an sdist and Kaggle hangs building it):

```bash
# Fixed env from the Kaggle log (torch 2.10.0+cu128):
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.10.0+cu128.html

# Or dynamic (same logic as notebook Cell 0, works across torch/CUDA):
python -c "import torch; pt=torch.__version__.split('+')[0]; cu=f\"cu{torch.version.cuda.replace('.', '')}\" if torch.version.cuda else 'cpu'; print(f'pip install torch-scatter -f https://data.pyg.org/whl/torch-{pt}+{cu}.html')"
```

## Usage — sequential co-dependent pipeline (same style as hybrid folder)

Each step consumes the previous step's artifact. One model per run:
default = clean, `--adversarial` = adv variant.

```bash
# Step 1 — build CPGs (Cell 1; trains NOTHING), saves {language}_cpg_bundle.pt
python main.py --language python
python main.py --language java
python main.py --language cpp --trial-samples 500 --limit 200

# Step 1b — adv pool + adv graphs ONLY (requires the clean bundle first;
# vocab/stats are reused untouched, clean outputs stay identical)
python main.py --language python --adversarial

# Step 2 — train exactly ONE model from the bundle (Cell 3 clean / Cell 4 adv)
python train.py --language python                # Model 1 clean
python train.py --language python --adversarial  # Model 2 adv (needs adv bundle)
python train.py --language cpp --adversarial --epochs 45
python train.py --language java --batch_size 32

# Grid search: every output-affecting hyperparameter is a flag (defaults = notebook values)
python train.py --language java --hidden_dim 128 --num_layers 2 --lr 1e-4 --epochs 60
python train.py --language python --dropout_gnn 0.2 --mask_rate 0.2 --patience 15

# Step 3 — test exactly ONE model (clean default, adv with flag)
python evaluate.py --language python
python evaluate.py --language cpp --adversarial

# 9-suite robustness benchmark, one model per run
python attack_evaluation.py --language python
python attack_evaluation.py --language java --adversarial

# Single-layer attacks, one model per run
python attack_authorship.py --language python --mode basic
python attack_statistical.py --language cpp --mode enhanced
python attack_semantic.py --language java --mode enhanced
python attack_full.py --language java --mode basic --target all
python attack_full.py --language python --mode enhanced --adversarial

# External OOD (SemEval-2026 A/B, HMCorp, GPTSniffer-java-only; dual-model, as in notebooks)
python external_eval.py --language python --suite all
python external_eval.py --language cpp --suite semeval_A
python external_eval.py --language java --suite gptsniffer

# Leakage audit (SHA256, <0.5% PASS; raw rows only, no graphs)
python leakage_audit.py --language python
```

Bundle files (created by `main.py`, consumed by everything else):
- `python_cpg_bundle.pt` / `java_cpg_bundle.pt` / `cpp_cpg_bundle.pt`
- hold vocab + BPE tokenizer + clean-locked normalization + graph lists
  (adv graphs present only after `main.py --adversarial`, which requires the
  clean bundle with matching `--trial-samples`/`--limit` and reuses its
  vocab/stats untouched)

Test sets: python/java evaluate on the FULL test set, C++ on the balanced
test set (rebuilt bundles required after this change; train/val pools and
checkpoints are unaffected).

Hyperparameter flags (`train.py`; omit any flag for the notebook default):
- Model: `--type_dim` 64, `--subword_dim` 128, `--hidden_dim` 256,
  `--num_layers` 4, `--dropout_gnn` 0.15,
  `--pool_hidden` 128, `--film_hidden` 128, `--cls_hidden1` 256,
  `--cls_hidden2` 64, `--dropout_cls1` 0.3, `--dropout_cls2` 0.2,
  `--mask_rate` 0.15
- Training: `--lr` 5e-4, `--weight_decay` 1e-3, `--epochs` 45,
  `--patience` 10, `--accum_steps` 2, `--tmax` 45, `--eta_min` 1e-6,
  `--smooth_pos` 0.975, `--smooth_neg` 0.025, `--grad_clip` 1.0,
  `--threshold` 0.50, `--batch_size` per-language default
- Locked (no flag; changing them breaks graph/checkpoint compat):
  `struct_dim=34`, `num_relations=16`, `global_dim=16` (16-d macro stats),
  conv kernel 3, BPE/subword sizes,
  `MAX_NODES`/`MAX_DEPTH`, augmentation rates, attack magnitudes
- Checkpoints store their `hyperparams`; eval/attack/external rebuild the
  model from them (old checkpoints fall back to notebook defaults)

Determinism: `main.py` and `train.py` both seed all RNGs with 42 first
(notebook Cell-1 state), so DataLoader shuffling, dropout, and token masking
follow the notebook trajectory and clean-test accuracy reproduces it.

Checkpoints (notebook-native, no `.npy` graph cache):
- `model_python_clean_baseline_checkpoint.pth` / `model_python_adv_augmented_checkpoint.pth`
- `model_cpp_clean_baseline_checkpoint.pth` / `model_cpp_adv_augmented_checkpoint.pth`
- `model_clean_baseline_checkpoint.pth` / `model_adv_augmented_checkpoint.pth` (java, no prefix — as in notebook)

Each `.pth` stores `epoch, model_state_dict, optimizer_state_dict, val_f1,
val_roc, optimal_threshold, normalization_means/stds, global_means/stds`.

## Layout

- `data_loader.py` — `balanced_subset` + magecode loading (shared, identical)
- `language_configs.py` — per-language RESERVED/TEXT_CAPTURE/SCOPE/triggers + shared RE/LAZY/MAGIC/BATCH defaults
- `graph_builder.py` — shared stylometry/macro/graph/normalization skeleton with per-language dispatch (outputs identical)
- `model.py` — `GatedGNNLayer` + `AdvancedASTGraphEncoder` (shared) + per-language save wrappers
- `attack_utils.py` — primitives + generators + eval/printing + distribution-shift audit
- `augment.py` — 20% all + 20% AI-only training augmentation (per-language exact)
- `pipeline.py` — Cell-1 pipeline shared by all CLIs
- `main.py` / `train.py` / `evaluate.py` — hybrid-style entry points, notebook-native artifacts
- `attack_evaluation.py` + `attack_authorship/statistical/semantic/full.py` — benchmarks
- `external_eval.py` — SemEval/HMCorp/GPTSniffer
- `leakage_audit.py` — SHA256 audit
