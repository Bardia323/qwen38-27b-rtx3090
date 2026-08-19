#!/bin/bash
# One-shot model preparation, idempotent: the README's Setup steps against
# /app/models (a bind mount / volume), each skipped when its result is already
# there. Runs on the CPU (no GPU needed). ~19.5 GB download + a few minutes of
# requantization; a fast-variant download of ~1 GB unless FAST_VARIANT=0.
#
#   docker compose run --rm prepare      (also runs automatically before single/batch)
set -e
cd /app
export PATH=/app/venv/bin:$PATH
BASE=${BASE_MODEL_DIR:-/app/models/Qwen3.8-27B-W4A16-AutoRound}
HF_REPO=${HF_REPO:-dbirks/Qwen3.8-27B-W4A16-AutoRound}

state() {  # prints the steps still to do
python - "$BASE" <<'EOF'
import json, os, sys
d = sys.argv[1].rstrip("/") + "/"
todo = []
if not (os.path.exists(d + "config.json") and os.path.exists(d + "model.safetensors.index.json")):
    print("download"); sys.exit()
idx = json.load(open(d + "model.safetensors.index.json"))["weight_map"]
if any(not os.path.exists(d + f) for f in set(idx.values())):
    print("download"); sys.exit()
if "lm_head.weight_packed" not in idx: todo.append("lm_head")
if not any(k.endswith("embed_tokens.weight_packed") for k in idx): todo.append("embed")
if "mtp.layers.0.mlp.down_proj.weight_packed" not in idx: todo.append("mtp")
if "mtp.draft_lm_head.weight_packed" not in idx or not os.path.exists(d + "mtp_draft_vocab_ids.pt"): todo.append("draft")
if os.environ.get("FAST_VARIANT", "1") != "0" and not os.path.exists(d[:-1] + "-fast/model.safetensors.index.json"):
    todo.append("fast")
print(" ".join(todo))
EOF
}

TODO=$(state)
if [ "$TODO" = "download" ]; then
  echo "== downloading $HF_REPO -> $BASE (~19.5 GB, resumable)"
  hf download "$HF_REPO" --local-dir "$BASE"
  TODO=$(state)
fi
[ "$TODO" = "download" ] && { echo "prepare: download incomplete (shards missing after hf download)"; exit 1; }
for step in $TODO; do
  case $step in
    lm_head) echo "== quant_lm_head.py (int8 lm_head)";      python quant_lm_head.py "$BASE" ;;
    embed)   echo "== quant_embed.py (int8 embeddings)";     python quant_embed.py "$BASE" ;;
    mtp)     echo "== quant_mtp.py (int8 MTP module)";       python quant_mtp.py "$BASE" ;;
    draft)   echo "== build_draft_vocab.py (40k draft head)"; python build_draft_vocab.py "$BASE" --ids draft_vocab_ids.json ;;
    fast)    echo "== fetch_fast_variant.py (int4-GPTQ lm_head/MTP + own-output draft vocab, ~1 GB)"
             python fetch_fast_variant.py "$BASE" "$BASE-fast" ;;
  esac
done
LEFT=$(state)
[ -z "$LEFT" ] || { echo "prepare: steps still missing after run: $LEFT"; exit 1; }
echo "prepare: model ready at $BASE$([ "${FAST_VARIANT:-1}" != 0 ] && echo " (+ $BASE-fast)")"
