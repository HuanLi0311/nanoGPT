#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
root=vision_model/data/imagenette_100
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60
hf download doyu/imagenette_224px_100 test.tgz --repo-type dataset --local-dir "$root"
tar -xzf "$root/test.tgz" -C "$root"
for class in "$root"/test/*; do
  name=$(basename "$class")
  mkdir -p "$root/train/$name" "$root/val/$name"
  for image in "$class"/*.jpg; do
    case "$image" in *8.jpg|*9.jpg) mv "$image" "$root/val/$name";; *) mv "$image" "$root/train/$name";; esac
  done
done
rm -rf "$root/test"
