from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("hydra")

from src.evaluation.checkpoint import (  # noqa: E402
    LEGACY_MODEL_TARGETS,
    load_checkpoint_config,
    load_model_config,
    migrate_model_config,
    repo_root,
)
from src.models.xjepa import (  # noqa: E402
    LEGACY_STATE_DICT_KEY_MAP,
    _remap_legacy_state_dict_keys,
)


CHECKPOINT_CONFIGS = {
    "clip": "src.models.contrastive_baselines.CLIP",
    "siglip": "src.models.contrastive_baselines.SigLIP",
    "xjepa_p": "src.models.xjepa.XJEPA_P",
    "xjepa_tc": "src.models.xjepa.XJEPA_TC",
    "xjepa_pa_lam003": "src.models.xjepa.XJEPA_PA",
    "xjepa_pa_lam01": "src.models.xjepa.XJEPA_PA",
    "xjepa_pa_lam03": "src.models.xjepa.XJEPA_PA",
    "xjepa_pa_lam10": "src.models.xjepa.XJEPA_PA",
}


@pytest.mark.parametrize("name,target", CHECKPOINT_CONFIGS.items())
def test_released_checkpoints_have_complete_model_configs(name, target):
    config = load_checkpoint_config(name)

    assert config is not None
    assert config["_target_"] == target
    assert (repo_root() / "configs" / "checkpoints" / f"{name}.yaml").exists()


@pytest.mark.parametrize(
    "name,alignment_lambda",
    [
        ("xjepa_p", 0.0),
        ("xjepa_tc", 0.0),
        ("xjepa_pa_lam003", 0.03),
        ("xjepa_pa_lam01", 0.1),
        ("xjepa_pa_lam03", 0.3),
        ("xjepa_pa_lam10", 1.0),
    ],
)
def test_released_xjepa_configs_pin_the_alignment_weight(name, alignment_lambda):
    config = load_checkpoint_config(name)

    assert config["alignment_lambda"] == pytest.approx(alignment_lambda)


def test_all_released_variants_share_one_cross_modal_predictor():
    for name in CHECKPOINT_CONFIGS:
        if not name.startswith("xjepa"):
            continue
        config = load_checkpoint_config(name)
        assert "predictor_sharing" not in config
        assert "vis_predictor" not in config
        assert "text_predictor" not in config


def test_bare_checkpoint_resolves_to_shipped_config(tmp_path):
    checkpoint = tmp_path / "xjepa_pa_lam01.ckpt"
    checkpoint.write_bytes(b"")

    model_cfg, run_cfg = load_model_config(checkpoint)

    assert model_cfg["_target_"] == "src.models.xjepa.XJEPA_PA"
    assert run_cfg["model"]["_target_"] == "src.models.xjepa.XJEPA_PA"


def test_sibling_model_config_takes_precedence(tmp_path):
    checkpoint = tmp_path / "clip.ckpt"
    checkpoint.write_bytes(b"")
    sibling = tmp_path / "clip.yaml"
    sibling.write_text(
        yaml.safe_dump({"_target_": "src.models.contrastive_baselines.SigLIP"}),
        encoding="utf-8",
    )

    model_cfg, _ = load_model_config(checkpoint)

    assert model_cfg["_target_"] == "src.models.contrastive_baselines.SigLIP"


def test_missing_config_raises(tmp_path):
    checkpoint = tmp_path / "unknown_model.ckpt"
    checkpoint.write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="Could not resolve a model config"):
        load_model_config(checkpoint)


def test_legacy_model_config_is_migrated():
    legacy = {
        "_target_": "src.models.paper.XJEPA_PA",
        "clip_loss": True,
        "contrastive_weight": 0.1,
        "objective_mode": "full_target",
        "predictor_sharing": "shared",
        "regularizer_mode": "none",
        "use_vicreg": False,
        "use_sigreg": False,
        "vicreg_dim": 512,
        "mlp_vicreg": True,
        "wd_scheduler": None,
        "log_grad_every": 500,
        "optimizer_groups": {
            "regularization_head": {"lr_mult": 10.0, "wd_mult": 1.0},
            "logit_scale": {"lr_mult": 10.0, "wd_mult": 0.0},
        },
    }

    migrated = migrate_model_config(legacy)

    assert migrated["_target_"] == LEGACY_MODEL_TARGETS["src.models.paper.XJEPA_PA"]
    assert migrated["projection_dim"] == 512
    assert "mlp_vicreg" not in migrated
    assert "objective_mode" not in migrated
    assert "wd_scheduler" not in migrated
    assert "log_grad_every" not in migrated
    assert migrated["optimizer_groups"]["projection_head"] == {
        "lr_mult": 10.0,
        "wd_mult": 1.0,
    }
    assert migrated["optimizer_groups"]["temperature"] == {"lr_mult": 10.0, "wd_mult": 0.0}


def test_legacy_state_dict_keys_are_renamed_in_place():
    state_dict = {
        "vicreg_proj_vis.net.weight": 1,
        "vicreg_proj_text.net.bias": 2,
        "vljepa_pred_proj.weight": 3,
        "vljepa_target_proj.weight": 4,
        "vljepa_logit_scale": 5,
        "vis_encoder.model.conv_proj.weight": 6,
        "vicreg_proj_vis_external.weight": 7,
    }

    _remap_legacy_state_dict_keys(state_dict, "")

    assert "global_proj_vis.net.weight" in state_dict
    assert "global_proj_text.net.bias" in state_dict
    assert "prediction_proj.weight" in state_dict
    assert "target_proj.weight" in state_dict
    assert state_dict["logit_scale"] == 5
    assert "vis_encoder.model.conv_proj.weight" in state_dict
    # Keys that merely share a prefix must not be touched.
    assert "vicreg_proj_vis_external.weight" in state_dict
    assert set(LEGACY_STATE_DICT_KEY_MAP) == {
        "vicreg_proj_vis",
        "vicreg_proj_text",
        "vljepa_pred_proj",
        "vljepa_target_proj",
        "vljepa_logit_scale",
        "vljepa_info_nce_loss",
    }


def _released_checkpoint_dir() -> Path | None:
    env = os.environ.get("XJEPA_CHECKPOINT_DIR")
    candidates = [Path(env).expanduser()] if env else []
    candidates.append(repo_root() / "checkpoints")
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "xjepa_pa_lam01.ckpt").exists():
            return candidate
    return None


@pytest.mark.integration
def test_released_checkpoints_load_strictly():
    """Load every published checkpoint into its refactored model.

    Skipped unless the checkpoints are present locally, e.g. after running
    ``bash scripts/download_checkpoints.sh``.
    """
    checkpoint_dir = _released_checkpoint_dir()
    if checkpoint_dir is None:
        pytest.skip("Set XJEPA_CHECKPOINT_DIR or download the released checkpoints")

    from src.evaluation.checkpoint import load_model_from_run

    expected_classes = {
        "xjepa_p": "XJEPA_P",
        "xjepa_tc": "XJEPA_TC",
        "xjepa_pa_lam01": "XJEPA_PA",
    }
    for name, class_name in expected_classes.items():
        model, _, _ = load_model_from_run(checkpoint_dir / f"{name}.ckpt", strict=True)
        assert type(model).__name__ == class_name
