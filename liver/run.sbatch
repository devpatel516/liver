#!/bin/bash
#SBATCH --job-name=liver
#SBATCH --partition=gpu            # EDIT: your GPU partition (run: sinfo -s)
##SBATCH --account=YOUR_ACCOUNT    # EDIT: only if your cluster requires it (remove one '#')
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00            # EDIT: your partition's max; keep WALLTIME_MIN in sync
#SBATCH --output=logs/%x_%j.out

# ---------------- EDIT THESE 3 LINES ----------------
MODULES="python/3.10 cuda/12.1"    # see: module avail   (use "" if not needed)
DATA=$HOME/liver/data/archive      # folder containing lits_*.csv and dataset_6/
WALLTIME_MIN=720                   # same as --time above, in minutes
# ----------------------------------------------------

cd "$SLURM_SUBMIT_DIR"
for m in $MODULES; do module load $m; done
source venv/bin/activate

if [ "$TEST" = "1" ]; then            # quick test:  TEST=1 sbatch run.sbatch
    PROC=test_processed; OUT=test_run; PREP="--limit_rows 400"; TRAIN="--epochs 2 --max_samples 600"
else
    PROC=processed;      OUT=run1;     PREP="";                 TRAIN="--epochs 250"
fi

[ -f $PROC/train.npz ] || python prepare.py --data_root "$DATA" --out_dir $PROC $PREP || exit 1
python -u train.py --data_dir $PROC --out_dir $OUT $TRAIN --time_limit_min $((WALLTIME_MIN - 15))
rc=$?

if [ $rc -eq 10 ] && [ "${N:-0}" -lt 20 ]; then          # ran out of time -> continue in a new job
    sbatch --export=ALL,N=$(( ${N:-0} + 1 )) run.sbatch
    exit 0
fi
exit $rc
