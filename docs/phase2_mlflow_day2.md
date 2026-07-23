# Phase 2 Day 2: MLflow Tracking Stack

Day 2 adds a durable local MLflow stack for experiment metadata and model
artifacts.

## Services

```text
training script -> MLflow tracking server -> Postgres metadata store
                                      `-> MinIO S3-compatible artifact store
```

- Postgres stores MLflow runs, params, metrics, and artifact metadata.
- MinIO stores checkpoint artifacts under the `mlflow-artifacts` bucket.
- `minio-create-bucket` creates the artifact bucket before MLflow starts.
- MLflow proxies artifact uploads, so local training clients only need the
  tracking URI.
- The training script reads `MLFLOW_TRACKING_URI` and
  `MLFLOW_EXPERIMENT_NAME` from the environment.

## Local Run

Create a local environment file:

```powershell
Copy-Item .env.example .env
```

Start the tracking stack:

```powershell
docker compose up -d postgres minio minio-create-bucket mlflow
```

Run a CPU smoke test against the tracking server:

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:MLFLOW_EXPERIMENT_NAME = "ddp-cifar10-platform"
python src/training/train_ddp.py --dataset fake --epochs 1 --batch-size 32 --limit-train-batches 5 --limit-val-batches 2 --device cpu
```

Open the MLflow UI at `http://localhost:5000`. The run should include params,
loss and accuracy metrics, throughput, and `checkpoints/latest.pt` as an
artifact.

Open the MinIO console at `http://localhost:9001` with the credentials from
`.env`. The `mlflow-artifacts` bucket should contain MLflow run artifacts.
