from pathlib import Path

import pytest

pytest.importorskip("hydra")

from hydra import compose, initialize


PAPER_EXPERIMENTS = {
    "VL4M_xjepa_p": ("src.models.xjepa.XJEPA_P", 0.0),
    "VL4M_xjepa_tc": ("src.models.xjepa.XJEPA_TC", 0.0),
    "VL4M_xjepa_pa_lam003": ("src.models.xjepa.XJEPA_PA", 0.03),
    "VL4M_xjepa_pa_lam010": ("src.models.xjepa.XJEPA_PA", 0.1),
    "VL4M_xjepa_pa_lam030": ("src.models.xjepa.XJEPA_PA", 0.3),
    "VL4M_xjepa_pa_lam100": ("src.models.xjepa.XJEPA_PA", 1.0),
}


@pytest.mark.parametrize("experiment,expected", PAPER_EXPERIMENTS.items())
def test_paper_pretraining_configs_are_explicit(experiment, expected):
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[f"experiment=pretraining/{experiment}"],
        )

    target, alignment_lambda = expected
    assert cfg.model._target_ == target
    assert cfg.model.alignment_lambda == pytest.approx(alignment_lambda)
    assert cfg.model.prediction_directions == ("i2t" if "tc" in experiment else "both")


def test_paper_experiment_files_exist():
    config_root = Path("configs/experiment/pretraining")
    for experiment in PAPER_EXPERIMENTS:
        assert (config_root / f"{experiment}.yaml").exists()
