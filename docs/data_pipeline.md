# VL4M pretraining data

All matched models are pretrained on the same cleaned image–text mixture
(`VL4M`) built from four public sources:

| Source | Pairs in the cleaned mixture |
|---|---:|
| Conceptual Captions (CC3M) | 2,919,397 |
| SBU Captions | 837,529 |
| MS-COCO Captions | 518,186 |
| Visual Genome | 754,356 |
| **Total** | **5,029,468** |

The counts match Appendix *Construction of the Pretraining Dataset*.  Two
filtering steps produce them:

1. **Path validity** — records whose image file cannot be resolved are dropped
   (SBU: 22,097 rows).
2. **Evaluation decontamination** — records whose source image appears in any
   evaluation benchmark used in the paper are dropped (COCO: 48,561, Visual
   Genome: 14,180, SBU: 113).

Decontamination covers the COCO Karpathy splits, the Flickr30k annotations
(linked through SBU Flickr URL ids), RefCOCO/RefCOCO+/RefCOCOg, SugarCrepe++,
VSR, SVO-Probes, and NLVR2.

## 1. Prepare the source data

Download the four sources and arrange the images below a single data root.  The
layout used for the paper is:

```text
<data_root>/
  cc3m/            # CC3M images + caption shard metadata
  coco/            # COCO images + annotations/karpathy_splits
  sbu/             # SBU images + captions
  vg/              # Visual Genome images + image_data.json
```

## 2. Build the cleaned manifest

The cleaned manifest is a JSON array with one object per image–caption pair:

```json
{"image_path": "cc3m/images/000000000.jpg", "caption": "a river has burst its banks"}
```

`image_path` is resolved relative to the data root.  The cleaning and
decontamination step is dataset-specific (source identifiers such as COCO image
ids, Flickr URL ids, and Visual Genome image ids are joined across sources), so
the script that produced the manifest is not part of this repository.  The
resulting manifest is the only input the public pipeline needs, and the check
below reproduces the reported mixture size:

```bash
python - <<'PY'
import json
rows = json.load(open("/data/4M/4M_cleaned.json"))
print(len(rows))          # 5029468
PY
```

## 3. Build the WebDataset shards

```bash
python scripts/build_vl4m_webdataset.py \
  --input-json /data/4M/4M_cleaned.json \
  --root-dir /data \
  --output-dir /data/4M_wds/train \
  --pattern 'vl4m-%06d.tar' \
  --maxcount 1000 \
  --maxsize-gb 1.0 \
  --expected-samples 5029468 \
  --verify-images
```

The builder writes one image plus one metadata JSON per sample:

```text
000000000.jpg
000000000.json
000000001.jpg
000000001.json
...
```

Each metadata record contains `idx` (row index in the cleaned manifest),
`image_path`, `caption`, and `source`.  `--expected-samples` makes the build
fail if the mixture size differs from the paper, `--verify-images` re-decodes
every image, and the report written to `<output-dir>/dataset.json` records the
source manifest SHA-256 plus per-shard sample counts.

With the paper settings this yields 5,030 shards, the last one holding 468
samples.

## 4. Pretrain

`configs/data/VL4M_wds.yaml` points at the shards and holds out a deterministic
10% validation split:

```yaml
train_shards: ${paths.data_dir}/4M_wds/train/vl4m-{000000..005029}.tar
metadata_path: ${paths.data_dir}/4M_wds/train/dataset.json
val_frac: 0.1
```

```bash
python src/train.py \
  experiment=pretraining/VL4M_xjepa_pa_lam010 \
  paths.data_dir=/data \
  trainer=ddp
```

`configs/data/VL4M.yaml` provides the equivalent JSON-backed datamodule, which
indexes `4M_cleaned.json` directly and is convenient for debugging:

```bash
python src/train.py \
  experiment=pretraining/VL4M_xjepa_pa_lam010 \
  data=VL4M \
  paths.data_dir=/data \
  trainer=ddp
```

Both datamodules emit the same sample contract
`(image, (caption, ""), -1, index, source)`; `src/data/collators.py` turns those
samples into the batch objects consumed by the models.  Pretraining images use
bicubic random resized crops to 224×224 with random horizontal flipping and
ImageNet normalization (`data.train_transform` in `configs/data/VL4M_wds.yaml`);
evaluation uses resize, center crop, and the same normalization.
