#!/bin/bash
# GPU1 work queue: chains behind the running palindrome+stack sweep.
set -u
PY=/mnt/ssd2/kda/venv/bin/python
export PYTHONPATH=/home/eitamar/code/kda
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=/mnt/data/kda/logs

echo "[q1] waiting for the palindrome/stack sweep... $(date +%H:%M)"
while pgrep -f '[t]asks palindrome,stack' >/dev/null; do sleep 60; done
echo "[q1] synthetic done $(date +%H:%M)"

echo "[q1] pretrain 1to1,7to1 $(date +%H:%M)"
$PY -u -m experiments.pretrain --sweep --ratios 1to1,7to1 --steps 3000 \
  --data-dir /mnt/ssd2/kda/data --out /mnt/data/kda/runs/pretrain \
  > $LOG/pretrain1.log 2>&1 && echo '[q1] pretrain ok' || echo '[q1] pretrain FAILED'

echo "[q1] all done $(date +%H:%M)"
