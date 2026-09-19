# Evaluation suite

Every evaluation freezes the pretrained model and reads it through a single
interface (`src/evaluation/adapters.py`).  Nothing is fine-tuned end to end.

## Tasks

| Alias | Experiment | Protocol | Reported metric |
|---|---|---|---|
| `coco_zeroshot` | `eval/coco_karpathy_zeroshot` | freeze, zero-shot | mean recall (I2T/T2I R@1,5,10) |
| `flickr30k_zeroshot` | `eval/flickr30k_zeroshot` | freeze, zero-shot | mean recall |
| `sugarcrepe_pp` | `eval/sugarcrepe_pp` | freeze, hard negatives | accuracy |
| `svo_probes` | `eval/svo_probes` | freeze, hard negatives | accuracy |
| `vsr` | `eval/vsr` | freeze, hard negatives | AUROC |
| `coco_adapter_tune` | `eval/coco_karpathy_adapter_tune` | frozen backbone + adapter | mean recall |
| `flickr30k_adapter_tune` | `eval/flickr30k_adapter_tune` | frozen backbone + adapter | mean recall |
| `nlvr2_token_interaction_probe` | `eval/nlvr2_token_interaction_probe` | frozen token probe | accuracy |
| `nlvr2_global_raw_probe` | `eval/nlvr2_global_raw_probe` | frozen pooled probe | accuracy |

## Running a single evaluation

```bash
apptainer exec --bind /path/to/data:/data ./x-jepa.sif python src/eval.py \
  experiment=eval/coco_karpathy_zeroshot \
  ckpt_path=checkpoints/xjepa_pa_lam01.ckpt \
  paths.data_dir=/data \
  trainer=gpu
```

`ckpt_path` accepts

* a released checkpoint file (`.../xjepa_pa_lam01.ckpt`), whose architecture is
  read from `configs/checkpoints/xjepa_pa_lam01.yaml`, or
* a training run directory containing `.hydra/config.yaml`.

## Running the paper suite

`scripts/quantitative_eval.py` drives the evaluations of the paper:

```bash
apptainer exec --bind /path/to/data:/data ./x-jepa.sif python scripts/quantitative_eval.py \
  --ckpt-path checkpoints/xjepa_pa_lam01.ckpt \
  --data-dir /data \
  --include paper_core \
  --output-dir results/xjepa_pa_lam01
```

Available aliases and groups:

| Group | Contents |
|---|---|
| `all` | all nine evaluations |
| `zero_shot` | COCO, Flickr30k, SugarCrepe++, VSR, SVO-Probes |
| `retrieval` | COCO/Flickr30k zero-shot and adapter tuning |
| `adapter` | COCO/Flickr30k adapter tuning |
| `probe` | both NLVR2 probes |
| `paper_core` | the main-results suite in paper order |

Overrides use `ALIAS:KEY=VALUE`, for example
`--eval-override vsr:trainer.devices=4`.  `--fast-debug` runs a single batch per
evaluation to check the pipeline.

## Protocols

**Zero-shot retrieval.**  Each test image and caption is encoded once, features
are normalized, and all image–text dot products are ranked in both directions.
Mean recall averages R@1, R@5 and R@10 over image-to-text and text-to-image
retrieval.  For X-JEPA [TC] the image-side feature is the predicted text-space
feature produced from image tokens and the text-side feature is the encoded
text-target feature.

**Retrieval adapters.**  The pretrained encoders stay frozen and in evaluation
mode; only an image adapter, a text adapter and a logit scale are trained.  Each
adapter is a residual 768→768 projection with LayerNorm whose residual branch is
initialized to zero, so the initial adapter is a no-op up to normalization.
Training uses 10 epochs, AdamW at 1e-3, weight decay 0.01, batch size 512 and
cosine annealing.  The best state by validation mean recall is evaluated on the
test split.

**Hard-negative diagnostics.**  SugarCrepe++ counts an example as correct only
when both positive captions score above the controlled hard negative.
SVO-Probes scores one sentence against a positive and a negative image.  VSR
scores spatial-relation compatibility and reports AUROC.  None of the three
trains task-specific parameters.

**NLVR2 probes.**  Both probes freeze the pretrained backbone and train only the
probe head for 10 epochs with AdamW at 1e-4, weight decay 0.01, dropout 0.1,
batch size 256 and cosine annealing; the best state by validation accuracy is
tested.  The global probe consumes pooled left/right/text features together with
their products and absolute differences.  The token-interaction probe consumes
raw image and text token sequences, spatially pooled by 2, through a 2-layer
Transformer with a learned CLS token; text padding is masked.

## Checkpoint selection

Checkpoint selection is fixed before final testing and uses validation retrieval
only: for each pretraining run select the checkpoint with the highest equally
weighted mean recall on COCO Karpathy validation and Flickr30k validation.  No
test split or language-sensitive benchmark is used.

```bash
apptainer exec --bind /path/to/data:/data ./x-jepa.sif python src/select_checkpoint.py \
  run_dir=logs/VL4M_xjepa_pa_lam010/runs/<timestamp> \
  paths.data_dir=/data
```

The selector writes `checkpoint_selection/selection.json` and
`checkpoint_selection/best_checkpoint.txt` inside the run directory.  Tasks,
weights and health thresholds are configurable in
`configs/checkpoint_selection/default.yaml`.

## Data layout

`paths.data_dir` must contain the evaluation datasets expected by
`configs/data/*.yaml`:

```text
<data_root>/
  coco/karpathy_splits/coco_karpathy_{train,val,test}.json
  coco/images/{train2017,val2017}
  flickr30k/annotations/flickr30k_{train,val,test}.json
  flickr30k/Images/
  scpp/                      # SugarCrepe++ jsonl files
  nlvr2/                     # NLVR2 annotations and images
  svo_probes/images/         # SVO-Probes images (annotations from the Hub)
```

`svo_probes` and `vsr` stream their annotations from the Hugging Face Hub
(`MichiganNLP/svo_probes`, `cambridgeltl/vsr_zeroshot`); `scripts/download_svo_probes.py`
fetches the SVO-Probes images that are not bundled with the dataset.
