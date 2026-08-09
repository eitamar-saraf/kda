#!/bin/bash
# GPU0 work queue: chains behind the running MQAR sweep.
set -u
PY=/mnt/ssd2/kda/venv/bin/python
export PYTHONPATH=/home/eitamar/code/kda
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=/mnt/data/kda/logs

echo "[q0] waiting for the mqar sweep... $(date +%H:%M)"
while pgrep -f '[t]asks mqar --seeds 0,1,2 ' >/dev/null; do sleep 60; done
echo "[q0] mqar done $(date +%H:%M)"

echo "[q0] 1/3 kernel benchmarks $(date +%H:%M)"
$PY -u -m experiments.bench_kernel --out /mnt/data/kda/runs/bench/kernels.json \
  > $LOG/bench.log 2>&1 && echo '[q0] bench ok' || echo '[q0] bench FAILED'

echo "[q0] 2/3 mqar solve-rate, 8 seeds at T=1024 $(date +%H:%M)"
$PY -u -m experiments.synthetic --sweep --tasks mqar \
  --variants kda,gated_deltanet,deltanet,mamba2,softmax \
  --seq-lens 1024 --seeds 0,1,2,3,4,5,6,7 \
  --out /mnt/data/kda/runs/solverate > $LOG/solverate.log 2>&1 \
  && echo '[q0] solverate ok' || echo '[q0] solverate FAILED'

echo "[q0] 3/3 pretrain attn_only,3to1,kda_only $(date +%H:%M)"
$PY -u -m experiments.pretrain --sweep --ratios attn_only,3to1,kda_only --steps 3000 \
  --data-dir /mnt/ssd2/kda/data --out /mnt/data/kda/runs/pretrain \
  > $LOG/pretrain0.log 2>&1 && echo '[q0] pretrain ok' || echo '[q0] pretrain FAILED'

echo "[q0] all done $(date +%H:%M)"
