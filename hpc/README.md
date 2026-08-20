# HPC workflow

Keep the Git checkout on a persistent project filesystem and put model caches and
experiment outputs on scratch/object storage. Do not commit weights, raw datasets,
checkpoints, prediction JSONL, or `.npz` artifacts.

## Setup

```bash
git clone git@github.com:shrn2/UQ_Compression.git
cd UQ_Compression
bash hpc/bootstrap.sh "$HOME/.venvs/upgd"
```

Set `HF_TOKEN` through the scheduler/account secret mechanism, not in a file committed to
Git. Set `UPGD_VENV`, `HF_HOME`, and `SCRATCH` in the job environment.

## Launching a sweep

```bash
mkdir -p logs
sbatch hpc/slurm/iter16_array.sbatch
```

The 24 array slots correspond to two models × four lambda values × three seeds. A run
should write a manifest containing the Git commit, config hash, host, GPU, CUDA/PyTorch
versions, and output paths. Jobs must be resumable and should skip completed artifacts.

## Reproducibility handoff

For every published result, preserve the commit SHA, YAML config, environment metadata,
Slurm job IDs, and compact `analysis_manifest.json`/`validation.json`. Store large outputs
under a versioned experiment directory such as
`$SCRATCH/upgd-runs/iter16_uncertainty_reg/<git-sha>/`.
