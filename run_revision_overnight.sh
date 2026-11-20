#!/bin/bash
# Revision benchmark run (npj review, Reviewer 1 comment 3). Run on an
# otherwise idle machine: every timed cell depends on it, and check_clean
# verifies it afterwards from the recorded CPU/wall ratios.
#
# What changed since the submitted run (results/sweep_pinned.jsonl, sweep_t8):
#   * Held-out accuracy. Every driver fits its readout on rows [200, 2000),
#     reports NRMSE on a validation segment [2000, 3000) and on a test segment
#     [3000, 4000); "nrmse" is now the test score. Accuracy is a validity
#     check (each timed reservoir does the task), not a result.
#   * Ridge lambda re-selected per (framework, task) on the validation segment
#     (tune.py --ridge-only); reservoir operating points are unchanged.
#   * Equal-work rows: brian2-1step and NEST-1step integrate at 1 ms, one step
#     per input sample like SPIRES, beside the idiomatic 0.1 ms rows.
#   * Thread scaling at 1, 2, 4 and 8 threads (2 and 4 at N = 500, 2000).
#
# Everything is written to results/revision/; the submitted results are not
# touched. Estimated duration ~6 h.
#
#   ./run_revision_overnight.sh                 # start now
#   at 23:00 -f run_revision_overnight.sh       # start later
cd "$(dirname "$0")"
PY=${BENCH_VENV:-$HOME/.pyvenv}/bin/python
OUT=results/revision
mkdir -p "$OUT"
say(){ echo "=== $1 : $(date '+%F %T') ==="; }

gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)
if [ "$gov" != performance ]; then
    echo "WARNING: governor is '$gov', expected 'performance'."
    echo "         Set it first: sudo cpupower frequency-set -g performance"
fi

# The submitted tuning is kept; the revision's lambdas go in config/tuned.json,
# which every runner reads.
[ -f config/tuned_submitted.json ] || cp config/tuned.json config/tuned_submitted.json

SIZES=100,200,500,1000,2000
TASKS=narma10,mackey_glass
ALL=SPIRES,reservoirpy,pytorch-esn,RC.jl,snnTorch,norse,rockpool,brian2,NEST,ANNarchy,brian2-1step,NEST-1step

say "ridge lambda on the validation segment (accuracy only, ~15 min)"
$PY src/bench/tune.py --ridge-only --n 300 --tasks $TASKS > $OUT/tune_ridge.log 2>&1

say "sweep 1 thread (12 frameworks x 5 sizes x 2 tasks x 10 reps)"
BENCH_THREADS=1 $PY src/bench/orchestrate.py --sizes $SIZES --reps 10 --tasks $TASKS \
    --frameworks $ALL --out $OUT/sweep_t1.jsonl > $OUT/sweep_t1.log 2>&1

sleep 20
say "sweep 8 threads"
BENCH_THREADS=8 $PY src/bench/orchestrate.py --sizes $SIZES --reps 10 --tasks $TASKS \
    --frameworks $ALL --out $OUT/sweep_t8.jsonl > $OUT/sweep_t8.log 2>&1

for t in 2 4; do
    sleep 20
    say "thread scaling: $t threads (N = 500, 2000)"
    BENCH_THREADS=$t $PY src/bench/orchestrate.py --sizes 500,2000 --reps 10 --tasks $TASKS \
        --frameworks $ALL --out $OUT/sweep_t$t.jsonl > $OUT/sweep_t$t.log 2>&1
done

for t in 1 8; do
    sleep 20
    say "brian2 runtime target, $t thread(s) (N = 500, NARMA-10)"
    BRIAN2_MODE=runtime BENCH_THREADS=$t $PY src/bench/orchestrate.py --sizes 500 --reps 10 \
        --tasks narma10 --frameworks brian2 --out $OUT/brian2_runtime_n500_t$t.jsonl \
        > $OUT/brian2_runtime_n500_t$t.log 2>&1
done

say "TIMING RUNS COMPLETE"
$PY - <<'EOF'
import sys; sys.path.insert(0, "src/bench")
from check_clean import check
check("revision/sweep_t1.jsonl", 1)
check("revision/brian2_runtime_n500_t1.jsonl", 1)
for t in (2, 4, 8):
    check(f"revision/sweep_t{t}.jsonl", t)
EOF
