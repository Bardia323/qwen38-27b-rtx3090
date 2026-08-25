#!/bin/bash
# Kill any leftover vLLM engine and wait for the card to actually release its memory.
#
# A vLLM engine that is mid-teardown (or wedged) keeps ~23 GB reserved; the next boot then
# fights it for memory during CUDA graph capture and hangs at 100% CPU, deaf to SIGTERM.
# Measured on this box: capture took 3s on a clean card, 542s against a dying engine, and
# the third attempt wedged outright. Draining first is what makes a restart reliable.
#
# This lives in a file rather than inline in qwen.ps1 because passing a multi-line script
# with embedded double quotes through `wsl.exe -- bash -c` loses the quoting: Windows
# argument splitting turned `pgrep -f "vllm serve"` into two arguments and bash died with
# `syntax error: unexpected end of file`.
if pgrep -f "vllm serve" >/dev/null; then
  pkill -f "vllm serve"; sleep 5
  pgrep -f "vllm serve" >/dev/null && { pkill -9 -f "vllm serve"; sleep 5; }
fi
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 2000 ] && break
  sleep 2
done
nvidia-smi --query-gpu=memory.used --format=csv,noheader
