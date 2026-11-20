"""Download an original public dataset into the server data directory."""

import argparse

from huggingface_hub import snapshot_download

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(snapshot_download(args.repo_id, repo_type="dataset", local_dir=args.output))
