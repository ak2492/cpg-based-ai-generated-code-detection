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
    --repo <owner>/<repo>

# Hybrid (+ scalers):
!python tools/hf_upload/upload_seed_models.py \
    --model-dir /kaggle/working/AI_Generated_Hybrid_Code_Detection \
    --repo <owner>/<repo> \
    --pattern '*_seed*.pt' --pattern '*_seed*.pkl'
```

Repo is always explicit via `--repo <owner>/<repo>` (no default); one
repo per language is the recommended convention. Missing repo, token, or
matching files skips the upload with a warning instead of an error.
