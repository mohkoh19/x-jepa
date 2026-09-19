import json

import pytest

torch = pytest.importorskip("torch")

from hydra import compose, initialize  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.evaluation.checkpoint_selection import (  # noqa: E402
    _task_cfg_with_name,
    aggregate_score,
    build_selection_payload,
    candidate_health_reasons,
    discover_checkpoints,
    evaluate_retrieval_task,
    rank_checkpoint_records,
    run_checkpoint_selection,
    score_candidate_record,
    write_selection_artifacts,
)


def _record(name, primary, mean_recall=0.0, modality_gap=1.0, epoch=0, warnings=0):
    return {
        "checkpoint": {
            "path": f"/tmp/{name}",
            "name": name,
            "epoch": epoch,
            "step": None,
        },
        "status": "accepted",
        "primary_score": primary,
        "aggregate_score": primary,
        "mean_recall": mean_recall,
        "modality_gap": modality_gap,
        "health": {
            "ok": True,
            "warnings": ["warn"] * warnings,
            "warning_count": warnings,
            "rejection_reasons": [],
        },
        "tasks": {},
    }


def test_checkpoint_discovery_orders_parsed_checkpoints_before_last(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    for name in ["last.ckpt", "epoch_010.ckpt", "epoch_005.ckpt", "epoch_005-step_100.ckpt"]:
        (checkpoint_dir / name).write_text("", encoding="utf-8")

    candidates = discover_checkpoints(run_dir)

    assert [candidate.name for candidate in candidates] == [
        "epoch_005.ckpt",
        "epoch_005-step_100.ckpt",
        "epoch_010.ckpt",
        "last.ckpt",
    ]
    assert candidates[-1].epoch is None


def test_checkpoint_discovery_can_scan_resumed_run_segments(tmp_path):
    early = tmp_path / "early"
    late = tmp_path / "late"
    (early / "checkpoints").mkdir(parents=True)
    (late / "checkpoints").mkdir(parents=True)
    (early / "checkpoints" / "epoch_001.ckpt").write_text("", encoding="utf-8")
    (early / "checkpoints" / "epoch_003.ckpt").write_text("", encoding="utf-8")
    (late / "checkpoints" / "epoch_005.ckpt").write_text("", encoding="utf-8")

    candidates = discover_checkpoints(late, additional_run_dirs=[early])

    assert [candidate.name for candidate in candidates] == [
        "epoch_001.ckpt",
        "epoch_003.ckpt",
        "epoch_005.ckpt",
    ]
    assert str(early.resolve()) in candidates[0].path
    assert str(late.resolve()) in candidates[-1].path


def test_checkpoint_ranking_uses_primary_score_outside_tolerance():
    ranked = rank_checkpoint_records(
        [
            _record("early.ckpt", primary=0.7, mean_recall=0.95, epoch=1),
            _record("late.ckpt", primary=0.8, mean_recall=0.1, epoch=2),
        ],
        primary_tolerance=0.001,
    )

    assert ranked[0]["checkpoint"]["name"] == "late.ckpt"


def test_checkpoint_ranking_uses_mean_recall_tie_breaker_inside_tolerance():
    ranked = rank_checkpoint_records(
        [
            _record("a.ckpt", primary=0.7000, mean_recall=0.90, epoch=1),
            _record("b.ckpt", primary=0.7005, mean_recall=0.80, epoch=2),
        ],
        primary_tolerance=0.001,
    )

    assert ranked[0]["checkpoint"]["name"] == "a.ckpt"


def test_checkpoint_health_rejects_nan_metrics_and_collapsed_embeddings():
    record = {
        "tasks": {
            "toy": {
                "metrics": {"i2t_r1": float("nan"), "t2i_r1": 1.0, "mean_recall": 1.0},
                "diagnostics": {
                    "all_finite": True,
                    "image_shape": [2, 2],
                    "text_shape": [2, 2],
                    "image_norm_min": 1.0,
                    "text_norm_min": 1.0,
                    "image_variance_mean": 0.0,
                    "text_variance_mean": 0.5,
                    "image_avg_cosine": 1.0,
                    "text_avg_cosine": 0.0,
                },
            }
        }
    }

    reasons, warnings = candidate_health_reasons(record, ["toy"])

    assert warnings == []
    assert any("i2t_r1" in reason for reason in reasons)
    assert any("variance collapsed" in reason for reason in reasons)
    assert any("average cosine" in reason for reason in reasons)


def test_candidate_scoring_averages_mean_bidirectional_r1():
    record = {
        "checkpoint": {"path": "/tmp/ckpt.ckpt", "name": "ckpt.ckpt", "epoch": 1, "step": None},
        "tasks": {
            "coco_val": {
                "metrics": {"i2t_r1": 0.2, "t2i_r1": 0.4, "mean_r1": 0.3, "mean_recall": 0.5},
                "diagnostics": {
                    "all_finite": True,
                    "image_shape": [2, 2],
                    "text_shape": [2, 2],
                    "image_norm_min": 1.0,
                    "text_norm_min": 1.0,
                    "image_variance_mean": 0.2,
                    "text_variance_mean": 0.2,
                    "image_avg_cosine": 0.0,
                    "text_avg_cosine": 0.0,
                    "modality_gap": 0.1,
                },
            },
            "flickr30k_val": {
                "metrics": {"i2t_r1": 0.6, "t2i_r1": 0.8, "mean_r1": 0.7, "mean_recall": 0.9},
                "diagnostics": {
                    "all_finite": True,
                    "image_shape": [2, 2],
                    "text_shape": [2, 2],
                    "image_norm_min": 1.0,
                    "text_norm_min": 1.0,
                    "image_variance_mean": 0.2,
                    "text_variance_mean": 0.2,
                    "image_avg_cosine": 0.0,
                    "text_avg_cosine": 0.0,
                    "modality_gap": 0.3,
                },
            },
        },
    }

    scored = score_candidate_record(record, ["coco_val", "flickr30k_val"])

    assert scored["status"] == "accepted"
    assert scored["primary_score"] == pytest.approx(0.5)
    assert scored["mean_recall"] == pytest.approx(0.7)
    assert scored["modality_gap"] == pytest.approx(0.2)


def test_aggregate_score_uses_configured_weights():
    record = {
        "tasks": {
            "coco_val": {"metrics": {"mean_recall": 0.5}},
            "flickr30k_val": {"metrics": {"mean_recall": 0.75}},
        }
    }

    score, missing = aggregate_score(
        record,
        {
            "coco_val.metrics.mean_recall": 0.50,
            "flickr30k_val.metrics.mean_recall": 0.50,
        },
    )

    assert missing == []
    assert score == pytest.approx(0.625)


def test_missing_optional_task_does_not_reject_when_allowed():
    record = {
        "checkpoint": {"path": "/tmp/ckpt.ckpt", "name": "ckpt.ckpt", "epoch": 1, "step": None},
        "tasks": {
            "coco_val": {
                "type": "retrieval",
                "metrics": {"i2t_r1": 1.0, "t2i_r1": 1.0, "mean_r1": 1.0, "mean_recall": 1.0},
                "diagnostics": {
                    "all_finite": True,
                    "image_shape": [2, 2],
                    "text_shape": [2, 2],
                    "image_norm_min": 1.0,
                    "text_norm_min": 1.0,
                    "image_variance_mean": 0.2,
                    "text_variance_mean": 0.2,
                    "image_avg_cosine": 0.0,
                    "text_avg_cosine": 0.0,
                    "modality_gap": 0.1,
                },
            },
            "flickr30k_val": {
                "type": "retrieval",
                "metrics": {},
                "error": "missing data",
            },
        },
    }

    scored = score_candidate_record(
        record,
        ["coco_val", "flickr30k_val"],
        aggregate_cfg={
            "allow_missing": True,
            "metric_weights": {
                "coco_val.metrics.mean_recall": 0.5,
                "flickr30k_val.metrics.mean_recall": 0.5,
            },
        },
    )

    assert scored["status"] == "accepted"
    assert scored["primary_score"] == pytest.approx(1.0)


def test_selection_artifacts_include_policy_and_best_checkpoint(tmp_path):
    ranked = [_record("best.ckpt", primary=0.9, mean_recall=0.8, epoch=3)]
    payload = build_selection_payload(
        run_dir=tmp_path,
        ranked_records=ranked,
        selector_cfg={"primary_tolerance": 0.001},
        tasks_cfg={"coco_val": {}, "flickr30k_val": {}},
        primary_tolerance=0.001,
    )

    output_path = write_selection_artifacts(payload, run_dir=tmp_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved["selection_policy"]["test_metrics_used"] is False
    assert saved["selected"]["checkpoint"]["name"] == "best.ckpt"
    assert (tmp_path / "checkpoint_selection" / "best_checkpoint.txt").read_text(
        encoding="utf-8"
    ).strip() == "/tmp/best.ckpt"
    assert (tmp_path / "checkpoint_selection" / "results.csv").exists()
    assert (tmp_path / "checkpoint_selection" / "results_long.csv").exists()
    assert "sugarcrepe_itt_acc" not in (
        tmp_path / "checkpoint_selection" / "results.csv"
    ).read_text(encoding="utf-8")
    assert "SugarCrepe" not in json.dumps(saved["selection_policy"])


class _ToyDataset:
    text = ["caption a", "caption b"]
    txt2img = {0: 0, 1: 1}
    img2txt = {0: [0], 1: [1]}


class _ToyDataModule:
    def setup(self, stage=None):
        self.stage = stage
        self.val_set = _ToyDataset()

    def val_dataloader(self):
        return [{"image": torch.eye(2)}]


class _ToyAdapter:
    def encode_texts(self, texts, normalize=True):
        del normalize
        lookup = {"caption a": torch.tensor([1.0, 0.0]), "caption b": torch.tensor([0.0, 1.0])}
        return torch.stack([lookup[text] for text in texts])

    def encode_images(self, images, normalize=True):
        del normalize
        return images


def test_retrieval_task_helper_computes_metrics_and_diagnostics():
    result = evaluate_retrieval_task(
        _ToyAdapter(),
        _ToyDataModule(),
        task_name="toy",
        metric_name="coco_karpathy",
        text_batch_size=1,
    )

    assert result["metrics"]["i2t_r1"] == pytest.approx(1.0)
    assert result["metrics"]["t2i_r1"] == pytest.approx(1.0)
    assert result["metrics"]["mean_r1"] == pytest.approx(1.0)
    assert result["diagnostics"]["all_finite"] is True
    assert result["num_images"] == 2
    assert result["num_texts"] == 2


def test_task_cfg_with_name_handles_structured_hydra_task_config():
    task_cfg = OmegaConf.create({"type": "retrieval", "metric": "coco_karpathy"})
    OmegaConf.set_struct(task_cfg, True)

    named = _task_cfg_with_name(task_cfg, "coco_val")

    assert named.name == "coco_val"
    assert named.type == "retrieval"
    assert named.metric == "coco_karpathy"
    assert "name" not in task_cfg


def test_checkpoint_selection_config_composes(monkeypatch):
    monkeypatch.setenv("PROJECT_ROOT", "/repo")
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="checkpoint_selection/default")

    assert cfg.selector.primary_tolerance == 0.001
    assert set(cfg.selector.tasks.keys()) == {"coco_val", "flickr30k_val"}
    assert cfg.selector.tasks.coco_val.datamodule.annotation_file.endswith(
        "coco_karpathy_val.json"
    )
    assert cfg.selector.tasks.flickr30k_val.datamodule.val_file == "flickr30k_val.json"
    assert cfg.selector.tasks.flickr30k_val.datamodule.test_file == "flickr30k_test.json"
    assert cfg.selector.tasks.flickr30k_val.datamodule.eval_split == "val"
    assert dict(cfg.selector.aggregate.metric_weights) == {
        "coco_val.metrics.mean_recall": 0.50,
        "flickr30k_val.metrics.mean_recall": 0.50,
    }
    assert "sugarcrepe" not in OmegaConf.to_yaml(cfg.selector).lower()
    forbidden = {
        "sugarcrepe_val.metrics.itt_acc",
        "coco_val.diagnostics.modality_gap_sanity",
        "flickr30k_val.diagnostics.modality_gap_sanity",
    }
    assert forbidden.isdisjoint(set(cfg.selector.aggregate.metric_weights.keys()))


def test_selector_evaluates_explicit_checkpoint_path_and_skips_ranking(monkeypatch, tmp_path):
    ckpt = tmp_path / "manual.ckpt"
    ckpt.write_text("", encoding="utf-8")
    calls = []

    def fake_evaluate(candidate, tasks_cfg, selector_cfg):
        calls.append(candidate.path)
        return _record(candidate.name, primary=0.8, mean_recall=0.7, epoch=candidate.epoch or 0)

    monkeypatch.setattr(
        "src.evaluation.checkpoint_selection.evaluate_checkpoint_candidate",
        fake_evaluate,
    )
    cfg = OmegaConf.create(
        {
            "run_dir": str(tmp_path),
            "selector": {
                "checkpoint_path": str(ckpt),
                "checkpoint_epoch": None,
                "write_csv": True,
                "output_subdir": "checkpoint_selection",
                "tasks": {"coco_val": {"type": "retrieval"}},
            },
        }
    )

    payload = run_checkpoint_selection(cfg)

    assert calls == [str(ckpt.resolve())]
    assert payload["selected"]["checkpoint"]["name"] == "manual.ckpt"
    assert (tmp_path / "checkpoint_selection" / "selection.json").exists()
    assert (tmp_path / "checkpoint_selection" / "results.csv").exists()


def test_selector_resolves_checkpoint_epoch(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "epoch_014-step_10.ckpt").write_text("", encoding="utf-8")
    (checkpoint_dir / "epoch_015-step_11.ckpt").write_text("", encoding="utf-8")
    seen = []

    def fake_evaluate(candidate, tasks_cfg, selector_cfg):
        seen.append(candidate)
        return _record(candidate.name, primary=0.8, mean_recall=0.7, epoch=candidate.epoch or 0)

    monkeypatch.setattr(
        "src.evaluation.checkpoint_selection.evaluate_checkpoint_candidate",
        fake_evaluate,
    )
    cfg = OmegaConf.create(
        {
            "run_dir": str(tmp_path),
            "selector": {
                "checkpoint_path": None,
                "checkpoint_epoch": 14,
                "write_csv": True,
                "output_subdir": "checkpoint_selection",
                "tasks": {"coco_val": {"type": "retrieval"}},
            },
        }
    )

    payload = run_checkpoint_selection(cfg)

    assert seen[0].name == "epoch_014-step_10.ckpt"
    assert payload["selected"]["checkpoint"]["epoch"] == 14


def test_run_checkpoint_selection_evaluates_all_resumed_segments(monkeypatch, tmp_path):
    early = tmp_path / "early"
    late = tmp_path / "late"
    (early / "checkpoints").mkdir(parents=True)
    (late / "checkpoints").mkdir(parents=True)
    (early / "checkpoints" / "epoch_001.ckpt").write_text("", encoding="utf-8")
    (late / "checkpoints" / "epoch_009.ckpt").write_text("", encoding="utf-8")
    seen = []

    def fake_evaluate(candidate, tasks_cfg, selector_cfg):
        del tasks_cfg, selector_cfg
        seen.append(candidate.path)
        return _record(
            candidate.name, primary=float(candidate.epoch or 0), epoch=candidate.epoch or 0
        )

    monkeypatch.setattr(
        "src.evaluation.checkpoint_selection.evaluate_checkpoint_candidate",
        fake_evaluate,
    )
    cfg = OmegaConf.create(
        {
            "run_dir": str(late),
            "selector": {
                "checkpoint_path": None,
                "checkpoint_epoch": None,
                "additional_run_dirs": [str(early)],
                "write_csv": True,
                "output_subdir": "checkpoint_selection",
                "tasks": {"coco_val": {"type": "retrieval"}},
            },
        }
    )

    payload = run_checkpoint_selection(cfg)

    assert len(seen) == 2
    assert any("epoch_001.ckpt" in path for path in seen)
    assert any("epoch_009.ckpt" in path for path in seen)
    assert payload["selected"]["checkpoint"]["name"] == "epoch_009.ckpt"
    assert payload["run_dir"] == str(late.resolve())
    assert payload["run_dirs"] == [str(early.resolve()), str(late.resolve())]
    assert (late / "checkpoint_selection" / "best_checkpoint.txt").exists()
