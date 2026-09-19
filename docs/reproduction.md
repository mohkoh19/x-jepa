# Reproducing the paper

This walkthrough takes a clean checkout to the numbers in the main table.  The
steps below were executed on the released checkpoints; the reported COCO
zero-shot mean recall of X-JEPA [P,A] λ=0.10 reproduces to 69.37 against 69.39
in the paper.

## 1. Environment

```bash
docker build -t x-jepa:latest .
apptainer build x-jepa.sif docker-daemon://x-jepa:latest
```

The container is used because the BERT tokenizer and the ImageNet-initialized
ViT are loaded through Hugging Face and TorchVision.  If your network requires a
proxy or an offline cache, pass it through:

```bash
apptainer exec \
  --bind /path/to/data:/data \
  --bind "$HOME/.cache/huggingface:/hf-cache" \
  --env HF_HOME=/hf-cache \
  ./x-jepa.sif bash
```

## 2. Checkpoints and data

```bash
bash scripts/download_checkpoints.sh          # -> checkpoints/*.ckpt
```

Arrange the evaluation data as described in
[`evaluation_suite.md`](evaluation_suite.md#data-layout) and point
`paths.data_dir` at the root.

## 3. Unit tests

```bash
apptainer exec ./x-jepa.sif python -m pytest tests -q     # CPU only
```

## 4. Single evaluation

```bash
apptainer exec --bind /data:/data ./x-jepa.sif python src/eval.py \
  experiment=eval/coco_karpathy_zeroshot \
  ckpt_path=checkpoints/xjepa_pa_lam01.ckpt \
  paths.data_dir=/data \
  trainer=gpu
```

Metrics appear both on stdout and in the run directory.  The released
checkpoints reproduce the reported COCO Karpathy values within tie-handling
noise:

| Metric | CLIP (reprod. / paper) | [P] | [TC] | [P,A] λ=0.10 |
|---|---:|---:|---:|---:|
| I2T R@1 | 52.74 / 52.78 | 0.02 / 0.02 | 21.74 / 21.74 | 56.84 / 56.90 |
| I2T R@5 | 80.60 / 80.42 | 0.12 / 0.12 | 42.44 / 42.44 | 81.64 / 81.68 |
| I2T R@10 | 88.28 / 88.24 | 0.12 / 0.12 | 50.46 / 50.46 | 88.98 / 88.98 |
| T2I R@1 | 38.91 / 38.72 | 0.01 / 0.01 | 27.28 / 27.29 | 39.90 / 39.90 |
| T2I R@5 | 68.38 / 68.09 | 0.11 / 0.11 | 56.35 / 56.35 | 69.28 / 69.29 |
| T2I R@10 | 79.23 / 79.06 | 0.23 / 0.22 | 69.30 / 69.31 | 79.58 / 79.57 |
| **Mean recall** | **68.02 / 67.89** | **0.10 / 0.10** | **44.60 / 44.60** | **69.37 / 69.39** |

Differences of at most a few tenths of a point are expected from tie handling
and floating-point ordering (a tenth of a point is five images out of 5000),
not from the model or the protocol.

## 5. Full main-results table

```bash
for ckpt in clip siglip xjepa_p xjepa_tc xjepa_pa_lam01; do
  apptainer exec --bind /data:/data ./x-jepa.sif python scripts/quantitative_eval.py \
    --ckpt-path "checkpoints/${ckpt}.ckpt" \
    --data-dir /data \
    --include paper_core \
    --output-dir "results/${ckpt}"
done
```

Each run writes one metrics file per evaluation plus a `summary.json` that can
be pasted next to the paper table.  The alignment ablation repeats step 5 with
`xjepa_pa_lam003`, `xjepa_pa_lam03`, and `xjepa_pa_lam10`.

The λ=0.03 checkpoint reproduces its ablation row as well:

```bash
apptainer exec --bind /data:/data ./x-jepa.sif python src/eval.py \
  experiment=eval/coco_karpathy_zeroshot \
  ckpt_path=checkpoints/xjepa_pa_lam003.ckpt \
  paths.data_dir=/data trainer=gpu
```

| Metric | Reproduced | Paper (Table 5) |
|---|---:|---:|
| COCO Karpathy zero-shot mean recall | 69.10 | 69.11 |
| Flickr30k zero-shot mean recall | 81.23 | 81.22 |

## 6. From scratch

Reproducing pretraining end to end requires the 5,029,468-pair VL4M mixture
([`data_pipeline.md`](data_pipeline.md)) and 8 NVIDIA H100 GPUs:

```bash
apptainer exec --bind /data:/data ./x-jepa.sif python src/train.py \
  experiment=pretraining/VL4M_xjepa_pa_lam010 \
  paths.data_dir=/data \
  trainer=ddp
```

`effective_batch_size=2048` fixes the global batch size; `data.batch_size` and
`trainer.devices`/`num_nodes` only decide how it is split, and the runtime
derives `trainer.accumulate_grad_batches` from the two.  Then select the
checkpoint by validation retrieval only:

```bash
apptainer exec --bind /data:/data ./x-jepa.sif python src/select_checkpoint.py \
  run_dir=logs/VL4M_xjepa_pa_lam010/runs/<timestamp> \
  paths.data_dir=/data
```
