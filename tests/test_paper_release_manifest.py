from pathlib import Path

import pytest
import yaml

from src.evaluation.checkpoint import load_checkpoint_config

MANIFEST = Path("configs/paper_release.yaml")
REQUIRED_BUILD_INPUTS = (
    ".project-root",
    "LICENSE",
    "README.md",
    "CITATION.cff",
    "pyproject.toml",
    "poetry.lock",
    "configs",
    "docs",
    "scripts",
    "src",
    "tests",
)


def _manifest() -> dict:
    return yaml.safe_load(MANIFEST.read_text())


def test_paper_release_manifest_matches_public_scope():
    manifest = _manifest()

    assert manifest["paper"]["dataset"]["expected_samples"] == 5_029_468
    assert len(manifest["models"]) == 8
    assert len(manifest["evaluations"]) == 9
    assert manifest["models"]["xjepa_tc"]["config"].endswith("VL4M_xjepa_tc")
    assert manifest["models"]["xjepa_pa_lam010"]["display_name"].endswith("0.10")


def test_every_released_checkpoint_has_a_config_and_an_experiment():
    manifest = _manifest()
    config_root = Path("configs")

    for entry in manifest["models"].values():
        checkpoint = entry["checkpoint"]
        experiment = config_root / f"{entry['config']}.yaml"
        model_config = config_root / "checkpoints" / f"{Path(checkpoint).stem}.yaml"

        assert experiment.exists(), f"missing pretraining config for {checkpoint}"
        assert model_config.exists(), f"missing architecture config for {checkpoint}"
        assert load_checkpoint_config(Path(checkpoint).stem) is not None


def test_paper_evaluations_have_experiment_configs():
    from scripts.quantitative_eval import EVAL_BY_ALIAS

    manifest = _manifest()
    config_root = Path("configs")

    for alias in manifest["evaluations"]:
        spec = EVAL_BY_ALIAS[alias]
        assert (config_root / "experiment" / f"{spec.experiment}.yaml").exists()


@pytest.mark.parametrize("alias", ["coco_zeroshot", "nlvr2_global_raw_probe"])
def test_representative_evaluations_compose(alias):
    pytest.importorskip("hydra")
    from hydra import compose, initialize

    from scripts.quantitative_eval import EVAL_BY_ALIAS

    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="eval/default",
            overrides=[f"experiment={EVAL_BY_ALIAS[alias].experiment}"],
        )

    assert cfg.model.ckpt_path is None  # resolved from the top-level `ckpt_path`
    assert cfg.evaluation.final_split == EVAL_BY_ALIAS[alias].final_split
    assert cfg.evaluation.protocol == EVAL_BY_ALIAS[alias].protocol


def _dockerignore_patterns() -> list[str]:
    return [
        line.strip()
        for line in Path(".dockerignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _excluded_by_dockerignore(relative_path: str) -> bool:
    """Approximate Docker's ignore rules for the patterns used in this repo."""
    from fnmatch import fnmatch

    parts = relative_path.split("/")
    # A pattern that matches a directory also excludes everything below it.
    candidates = ["/".join(parts[: index + 1]) for index in range(len(parts))]
    for pattern in _dockerignore_patterns():
        if pattern.startswith("!"):
            continue
        needle = pattern.lstrip("/")
        for candidate in candidates:
            if fnmatch(candidate, needle) or fnmatch(candidate, needle.rstrip("/")):
                return True
    return False


def test_container_build_inputs_are_not_ignored():
    for name in REQUIRED_BUILD_INPUTS:
        assert Path(name).exists(), f"Dockerfile copies {name}, which is missing"

    tracked = [
        path
        for path in Path("src").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    assert tracked, "expected source files in src/"
    for path in tracked:
        assert not _excluded_by_dockerignore(str(path)), f"{path} would be excluded from the image"
