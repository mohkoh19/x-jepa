# Released checkpoint configs

Each YAML file in this directory describes the architecture of one checkpoint
in the Hugging Face repository
[`mohkoh/x-jepa`](https://huggingface.co/mohkoh/x-jepa).  The file name matches
the checkpoint file name, so

```bash
python src/eval.py experiment=eval/coco_karpathy_zeroshot \
  ckpt_path=checkpoints/xjepa_pa_lam01.ckpt
```

rebuilds the architecture from `configs/checkpoints/xjepa_pa_lam01.yaml` and
loads the released weights into it.  The configs only describe the model; the
datasets and evaluation protocols come from the `experiment=eval/...` configs.

The files are exported from the matching pretraining experiment so that the
released architecture is explicit rather than inherited.  To regenerate one:

```python
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import yaml

with initialize_config_dir(version_base="1.3", config_dir="configs"):
    cfg = compose(
        config_name="train.yaml",
        overrides=["experiment=pretraining/VL4M_xjepa_pa_lam010"],
    )

with open("configs/checkpoints/xjepa_pa_lam01.yaml", "w") as handle:
    yaml.safe_dump(OmegaConf.to_container(cfg.model, resolve=True), handle, sort_keys=False)
```
