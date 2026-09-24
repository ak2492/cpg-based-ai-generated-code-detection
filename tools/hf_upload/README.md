# HF upload helper (shared by CPG + Hybrid folders)

Uploads per-seed model files (filenames embed language + seed + variant)
from a Kaggle working directory to a Hugging Face model repo.

Token: `HF_TOKEN` in Kaggle Secrets (or env var). Nothing secret is written
to disk. The script never deletes anything; `create_repo(exist_ok=True)`
reuses an existing repo.

```bash
!pip -q install -U huggingface_hub
!python tools/hf_upload/upload_seed_models.py \
    --model-dir /kaggle/working/cpg-based-ai-generated-code-detection \
    --repo ak2492/cpg_based_detection-models_cpp

# Hybrid (+ scalers):
!python tools/hf_upload/upload_seed_models.py \
    --model-dir /kaggle/working/AI_Generated_Hybrid_Code_Detection \
    --repo ak2492/hybrid_detection-models_python \
    --pattern '*_seed*.pt' --pattern '*_seed*.pkl'
```

Repo convention: one repo per language —
`ak2492/cpg_based_detection-models_{cpp|python|java}`,
`ak2492/hybrid_detection-models_{cpp|python|java}`.
