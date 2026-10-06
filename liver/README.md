# Liver ResUNet on a cluster — 5 steps

Files: `prepare.py` (data → arrays), `train.py` (model + training + test results), `run.sbatch` (the job), `requirements.txt`.

## 1. On your PC (PowerShell): pack and upload
```powershell
cd $HOME\Downloads
tar -czf archive.tar.gz archive
scp archive.tar.gz  USER@LOGIN_HOST:~/
scp -r C:\path\to\liver  USER@LOGIN_HOST:~/liver      # this folder
```
(WinSCP works too and can resume a broken upload.)

## 2. On the cluster: unpack and install (once)
```bash
ssh USER@LOGIN_HOST
cd ~/liver
mkdir -p data logs && tar -xzf ~/archive.tar.gz -C data      # -> ~/liver/data/archive
module avail python cuda                                      # note the names
module load python/3.10 cuda/12.1                             # use the names you saw
python3 -m venv venv && source venv/bin/activate
pip install torch && pip install -r requirements.txt
```

## 3. Edit `run.sbatch` (top 3 blocks only)
`--partition` (find with `sinfo -s`), `--time`, `MODULES`, `DATA`, `WALLTIME_MIN`.

## 4. Quick test (~10 min), then the real run
```bash
TEST=1 sbatch run.sbatch        # 2 epochs on a tiny subset; look for "TEST RESULTS" in the log
sbatch run.sbatch               # full run: prepares data, trains 250 epochs, evaluates
```
Watch it: `squeue -u $USER` and `tail -f logs/liver_*.out`. Cancel: `scancel JOBID`.
If the time limit hits, the job saves and re-submits itself and continues automatically.

## 5. Download results
Everything is in `run1/` (`best.pt`, `train_log.csv`, `results/metrics.json`, `results/sample_*.png`):
```powershell
scp -r USER@LOGIN_HOST:~/liver/run1/results .\results
scp USER@LOGIN_HOST:~/liver/run1/best.pt .
```

## If something fails
| Error | Fix |
|---|---|
| invalid partition / account | fix `--partition` / `--account` in run.sbatch |
| `Could not auto-detect image/mask columns` | add `--img_col NAME --mask_col NAME` to the `prepare.py` line |
| `No CSV rows resolved` | `DATA` must be the folder that contains `dataset_6/` |
| CUDA out of memory | in `train.py` run line add `--batch_size 16` |
| Want later epochs to keep learning | add `--lr_decay 0.1` to the `train.py` line (paper's 5e-3 freezes learning after epoch 50) |
