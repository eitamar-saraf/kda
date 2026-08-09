#!/bin/bash
# Idempotent restart of the whole GPU pipeline on trex.
#
# Safe to run when jobs are already up: every launch is guarded by a pgrep, so a
# network partition that left the work running does not get a second copy fighting
# for the same GPU. The sweeps themselves are resumable -- each run writes its own
# JSON and existing ones are skipped -- so a reboot costs at most one run.
set -u
PY=/mnt/ssd2/kda/venv/bin/python
export PYTHONPATH=/home/eitamar/code/kda
LOG=/mnt/data/kda/logs
mkdir -p "$LOG" /mnt/data/kda/runs

start() {   # start <pgrep-pattern> <logfile> <gpu> <args...>
  local pat="$1" logf="$2" gpu="$3"; shift 3
  if pgrep -f "$pat" >/dev/null; then
    echo "[resume] already running: $pat"
    return
  fi
  echo "[resume] starting on gpu$gpu: $*"
  CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    setsid nohup "$PY" -u "$@" </dev/null >>"$logf" 2>&1 &
  disown
}

start '[t]asks mqar --seeds 0,1,2 ' "$LOG/sw_mqar.log" 0 \
  -m experiments.synthetic --sweep --tasks mqar --seeds 0,1,2 \
  --out /mnt/data/kda/runs/synthetic

start '[t]asks palindrome,stack' "$LOG/sw_palstack.log" 1 \
  -m experiments.synthetic --sweep --tasks palindrome,stack --seeds 0,1,2 \
  --out /mnt/data/kda/runs/synthetic

for q in 0 1; do
  if pgrep -f "[q]ueue_gpu${q}.sh" >/dev/null; then
    echo "[resume] queue$q already running"
  else
    echo "[resume] starting queue$q"
    setsid nohup /mnt/data/kda/queue_gpu${q}.sh </dev/null >>"$LOG/queue${q}.log" 2>&1 &
    disown
  fi
done
sleep 3
echo "[resume] processes now:"
pgrep -af '[e]xperiments\.|[q]ueue_gpu' | sed 's|.*/mnt/ssd2/kda/venv/bin/python -u ||;s|.*bash ||' | sort -u
