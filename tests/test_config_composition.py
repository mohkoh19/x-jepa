import pytest

pytest.importorskip("hydra")

from hydra import compose, initialize


@pytest.mark.parametrize(
    "experiment,target",
    [
        ("VL4M_clip", "src.models.contrastive_baselines.CLIP"),
        ("VL4M_siglip", "src.models.contrastive_baselines.SigLIP"),
        ("VL4M_xjepa_p", "src.models.xjepa.XJEPA_P"),
        ("VL4M_xjepa_tc", "src.models.xjepa.XJEPA_TC"),
        ("VL4M_xjepa_pa_lam003", "src.models.xjepa.XJEPA_PA"),
        ("VL4M_xjepa_pa_lam010", "src.models.xjepa.XJEPA_PA"),
        ("VL4M_xjepa_pa_lam030", "src.models.xjepa.XJEPA_PA"),
        ("VL4M_xjepa_pa_lam100", "src.models.xjepa.XJEPA_PA"),
    ],
)
def test_paper_pretraining_configs_compose_to_paper_models(experiment, target):
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[f"experiment=pretraining/{experiment}"],
        )

    assert cfg.model._target_ == target
    assert cfg.model.warmup == 4
    assert cfg.model.ipe_scale == pytest.approx(2.0)
