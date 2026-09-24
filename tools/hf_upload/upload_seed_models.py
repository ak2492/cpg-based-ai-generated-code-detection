"""Upload per-seed model files from Kaggle to Hugging Face.

Shared by both folders (CPG .pth checkpoints and Hybrid .pt/.pkl files):
per-seed filenames already embed language + seed + variant, e.g.
  model_python_clean_baseline_checkpoint_seed43.pth
  python_best_model_seed43.pt

The HF token is read from Kaggle Secrets (HF_TOKEN) or the HF_TOKEN
environment variable; it is never written to disk. Assumes the secret
already exists (user uploads it once in the Kaggle notebook settings).

Kaggle usage:
  !pip -q install -U huggingface_hub
  !python tools/hf_upload/upload_seed_models.py \\
      --model-dir /kaggle/working/cpg-based-ai-generated-code-detection \\
      --repo <owner>/<repo>

Local dry run (lists matches without uploading):
  python tools/hf_upload/upload_seed_models.py --model-dir . \\
      --repo <owner>/<repo> --dry-run

Missing --repo, token, or matching files skips the upload with a warning
instead of an error; nothing here ever raises for a skipped upload.
"""
import argparse
import glob
import os


def get_hf_token():
    # I try Kaggle Secrets first, then the environment, then the cached
    # huggingface_hub login, because I want one script for both worlds.
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            return token
    except Exception:
        pass
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import whoami
        whoami()
        return True  # cached login will be used
    except Exception:
        return None


def collect_files(model_dir, patterns, include_canonical=False):
    found = []
    for pat in patterns:
        found.extend(glob.glob(os.path.join(model_dir, pat)))
    if not include_canonical:
        found = [f for f in found if "_seed" in os.path.basename(f)]
    # I dedupe here because overlapping --pattern values match the same file.
    return sorted(set(found))


def upload_models(model_dir, repo_id, patterns=("*_seed*.pth",), token=None,
                  dry_run=False, include_canonical=False,
                  commit_prefix="Upload"):
    # I check every skip condition before importing huggingface_hub so a
    # missing dependency also degrades to a warned skip, never a crash.
    if not repo_id:
        print("[!] upload skipped: no --repo <owner/name> given.")
        return []
    files = collect_files(model_dir, patterns, include_canonical)
    print("Models found:")
    for f in files:
        print(f"  {os.path.basename(f)}  ->  {os.path.getsize(f) / (1024 ** 3):.2f} GB")
    if not files:
        print(f"[!] upload skipped: no model files in {model_dir} for {patterns}.")
        return []
    if dry_run:
        print("[dry-run] nothing uploaded.")
        return files

    token = token or get_hf_token()
    if not token:
        print("[!] upload skipped: no HF token (add HF_TOKEN to Kaggle Secrets or env).")
        return []
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[!] upload skipped: huggingface_hub not installed.")
        return []
    uploaded = []
    try:
        api = HfApi(token=(token if isinstance(token, str) else None))
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"\nRepository: https://huggingface.co/{repo_id}")
        for local_path in files:
            filename = os.path.basename(local_path)
            print(f"\nUploading: {filename}")
            try:
                api.upload_file(path_or_fileobj=local_path, path_in_repo=filename,
                                repo_id=repo_id, repo_type="model",
                                commit_message=f"{commit_prefix} {filename}")
            except Exception as e:
                print(f"[!] upload failed for {filename}: {e} (continuing)")
                continue
            print(f"Done: {filename}")
            uploaded.append(local_path)
    except Exception as e:
        print(f"[!] upload aborted: {e}")
        return uploaded
    print("\n========================================")
    print("ALL MODELS UPLOADED SUCCESSFULLY")
    print(f"https://huggingface.co/{repo_id}")
    print("========================================")
    return uploaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload per-seed models to Hugging Face")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--repo", type=str, default=None,
                        help="Full repo id 'owner/name' (missing repo skips gracefully)")
    parser.add_argument("--pattern", action="append", default=["*_seed*.pth"],
                        help="Repeatable glob (default: *_seed*.pth). "
                             "Add --pattern '*_seed*.pkl' for Hybrid scalers.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-canonical", action="store_true",
                        help="Also upload files without '_seed' in the name")
    args = parser.parse_args()
    # I always exit 0 here because skipped uploads are warnings, not errors.
    upload_models(args.model_dir, args.repo, tuple(args.pattern),
                  dry_run=args.dry_run,
                  include_canonical=args.include_canonical)
