# Operations Runbook

This runbook keeps the local platform checks short and command-driven. Generated
data, checkpoints, reports, and runtime state should stay outside git.

## Startup

Install dependencies:

```bash
pip install -r requirements.txt
```

Start MLflow dependencies:

```bash
docker compose up -d postgres minio minio-create-bucket mlflow
docker compose ps
```

Run a small training job without external downloads:

```bash
python src/training/train_ddp.py --dataset fake --epochs 1 --batch-size 32 --limit-train-batches 2 --limit-val-batches 1 --device cpu --disable-mlflow
```

Serve a checkpoint:

```bash
export MODEL_CHECKPOINT_PATH=checkpoints/latest.pt
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

On Windows PowerShell:

```powershell
$env:MODEL_CHECKPOINT_PATH = "checkpoints/latest.pt"
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

Generate a monitoring report:

```bash
python src/monitoring/build_reference.py
python src/monitoring/generate_drift_report.py
```

## Health Checks

Docker Compose services:

```bash
docker compose ps
```

Serving process:

```bash
curl http://localhost:8000/live
```

Model load status:

```bash
curl http://localhost:8000/health
```

Prediction path:

```bash
python scripts/smoke_predict.py
```

Full local smoke path:

```bash
python scripts/platform_smoke.py
```

Automated tests:

```bash
pytest -q
```

## Common Failure Checks

If Docker services are unavailable:

```bash
docker compose ps
docker compose logs mlflow
docker compose logs postgres
docker compose logs minio
```

If serving health reports `not_loaded`, check the configured checkpoint path:

```bash
echo $MODEL_CHECKPOINT_PATH
ls checkpoints
```

On Windows PowerShell:

```powershell
$env:MODEL_CHECKPOINT_PATH
Get-ChildItem checkpoints
```

If training cannot download CIFAR-10, use the fake-data smoke path first:

```bash
python src/training/train_ddp.py --dataset fake --epochs 1 --batch-size 32 --limit-train-batches 2 --limit-val-batches 1 --device cpu --disable-mlflow
```

If monitoring reports are missing, regenerate local artifacts:

```bash
python src/monitoring/build_reference.py
python src/monitoring/generate_drift_report.py
```

## Shutdown

Stop local infrastructure:

```bash
docker compose down
```

Remove generated local artifacts only when they are no longer needed:

```bash
rm -rf outputs/
```

On Windows PowerShell:

```powershell
Remove-Item -Recurse -Force outputs
```

## Repo Hygiene

Check source changes and ignored local artifacts:

```bash
git status --short
git status --short --ignored
```

Expected ignored paths include:

```text
.env
data/
checkpoints/
mlruns/
outputs/
.runtime/
```
