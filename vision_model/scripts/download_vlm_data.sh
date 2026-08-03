#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_ETAG_TIMEOUT=60

download() {
  .venv/bin/hf download --repo-type dataset "$1" --local-dir "$2" --max-workers 4 --quiet
}

root=vision_model/data/vlm
mkdir -p "$root"
download jxie/flickr8k "$root/flickr8k"
download naver-clova-ix/cord-v2 "$root/cord_v2"
