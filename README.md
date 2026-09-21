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

```bash
pip install -r requirements.txt
# torch-scatter wheel matching your torch/CUDA (same as notebook Cell 0):
#   pt_version=$(python -c "import torch; print(torch.__version__.split('+')[0])")
#   pip install torch-scatter -f https://data.pyg.org/whl/torch-${pt_version}+cu121.html
```

## Usage (same style as hybrid folder)

```bash
# Full end-to-end per language (data -> graphs -> clean + adv training)
python main.py --language python
python main.py --language java
python main.py --language cpp

# Single-model training (mirrors train.py in hybrid folder)
python train.py --language python
python train.py --language cpp --adversarial
python train.py --language java --epochs 45 --batch_size 32

# Clean test evaluation (mirrors evaluate.py in hybrid folder)
python evaluate.py --language python
python evaluate.py --language cpp --adversarial

# 9-suite robustness benchmark (clean + auth/stat/sem/full x basic/enhanced)
python attack_evaluation.py --language python
python attack_evaluation.py --language java --limit 200

# Single-layer attacks
python attack_authorship.py --language python --mode basic
python attack_statistical.py --language cpp --mode enhanced
python attack_semantic.py --language java --mode enhanced
python attack_full.py --language java --mode basic --target all

# External OOD (SemEval-2026 A/B, HMCorp, GPTSniffer-java-only)
python external_eval.py --language python --suite all
python external_eval.py --language cpp --suite semeval_A
python external_eval.py --language java --suite gptsniffer

# Leakage audit (SHA256, <0.5% PASS)
python leakage_audit.py --language python
```

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
