from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import src.eval as eval_entrypoint
from src.evaluation.checkpoint_selection import evaluate_zero_shot_task
from src.evaluation.logging import log_evaluation_metrics


class FakeTrainer:
    def __init__(self):
        self.logger = None
        self.loggers = []
        self.callback_metrics = {"fallback": torch.tensor(0.5)}
        self.is_global_zero = True
        self.fit_ckpt_path = None
        self.validate_ckpt_path = None

    def fit(self, model, datamodule=None, ckpt_path=None):
        self.fit_ckpt_path = ckpt_path

    def validate(self, model, datamodule=None, ckpt_path=None):
        self.validate_ckpt_path = ckpt_path
        return [{"metric": torch.tensor(1.0)}]

    def test(self, model, datamodule=None, ckpt_path=None):
        return [{"test_metric": torch.tensor(2.0)}]


def _cfg(tmp_path, **evaluation_overrides):
    evaluation = {
        "fit": False,
        "validate": True,
        "test": False,
        "lightning_ckpt_path": None,
        "log_hparams": False,
        "metric_prefix": "eval",

    }
    evaluation.update(evaluation_overrides)
    return OmegaConf.create(
        {
            "ckpt_path": "/pretrain/run",
            "seed": None,
            "paths": {"output_dir": str(tmp_path)},
            "logger": None,
            "evaluation": evaluation,
        }
    )


def test_eval_does_not_pass_top_level_ckpt_to_validate(monkeypatch, tmp_path):
    trainer = FakeTrainer()
    monkeypatch.setattr(eval_entrypoint, "prepare_runtime", lambda cfg: None)
    monkeypatch.setattr(
        eval_entrypoint,
        "build_evaluation_context",
        lambda cfg: {
            "datamodule": object(),
            "model": SimpleNamespace(),
            "trainer": trainer,
        },
    )

    eval_entrypoint.evaluate(_cfg(tmp_path, lightning_ckpt_path="/optional/eval-module.ckpt"))

    assert trainer.validate_ckpt_path == "/optional/eval-module.ckpt"
    assert trainer.validate_ckpt_path != "/pretrain/run"


def test_eval_validates_current_probe_weights_after_fit(monkeypatch, tmp_path):
    trainer = FakeTrainer()
    monkeypatch.setattr(eval_entrypoint, "prepare_runtime", lambda cfg: None)
    monkeypatch.setattr(
        eval_entrypoint,
        "build_evaluation_context",
        lambda cfg: {
            "datamodule": object(),
            "model": SimpleNamespace(),
            "trainer": trainer,
        },
    )

    eval_entrypoint.evaluate(_cfg(tmp_path, fit=True, lightning_ckpt_path="/resume/probe.ckpt"))

    assert trainer.fit_ckpt_path == "/resume/probe.ckpt"
    assert trainer.validate_ckpt_path is None


def test_final_metrics_are_prefixed_and_logged_without_hparams(tmp_path):
    class FakeLogger:
        def __init__(self):
            self.metrics = None
            self.hparams_called = False

        def log_metrics(self, metrics):
            self.metrics = metrics

        def log_hyperparams(self, params):
            self.hparams_called = True

    logger = FakeLogger()
    trainer = SimpleNamespace(
        logger=logger,
        loggers=[logger],
        is_global_zero=True,
    )
    cfg = _cfg(tmp_path, metric_prefix="eval")

    logged = log_evaluation_metrics({"itt_acc": torch.tensor(0.75)}, cfg, trainer)

    assert logged == {"eval/itt_acc": 0.75}
    assert logger.metrics == logged
    assert logger.hparams_called is False
    assert (tmp_path / "evaluation_metrics.json").exists()


def test_zero_shot_dispatcher_rejects_unknown_task_type():
    with pytest.raises(ValueError, match="Unsupported zero-shot task type `custom`"):
        evaluate_zero_shot_task(
            adapter=SimpleNamespace(resolved_ckpt="/tmp/model.ckpt"),
            task_cfg={
                "name": "custom_val",
                "type": "custom",
            },
            selector_cfg={},
        )
