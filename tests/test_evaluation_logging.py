from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from src.evaluation.logging import log_evaluation_metrics
from src.evaluation.nlvr2 import NLVR2Evaluator
from src.evaluation.retrieval import compute_retrieval_metrics


def test_retrieval_metrics_are_relative_to_eval_prefix(tmp_path):
    image_features = torch.eye(2)
    text_features = torch.eye(2)
    metrics = compute_retrieval_metrics(
        image_features=image_features,
        text_features=text_features,
        txt2img={0: 0, 1: 1},
        img2txt={0: [0], 1: [1]},
    )
    cfg = SimpleNamespace(
        get=lambda key, default=None: {
            "ckpt_path": "/pretrain/run",
            "evaluation": {"metric_prefix": "eval/coco-retrieval"},
            "paths": SimpleNamespace(output_dir=str(tmp_path)),
        }.get(key, default),
        evaluation={"metric_prefix": "eval/coco-retrieval"},
        paths=SimpleNamespace(output_dir=str(tmp_path)),
    )
    trainer = SimpleNamespace(logger=None, loggers=[], is_global_zero=True)

    logged = log_evaluation_metrics(metrics, cfg, trainer)

    assert "eval/coco-retrieval/i2t_r1" in logged
    assert "eval/coco-retrieval/retrieval/i2t_r1" not in logged


def test_nlvr2_metric_keys_are_flattened_and_train_uses_train_root():
    model = NLVR2Evaluator(ckpt_path="", adapter=object())

    assert model._metric_key("train", "loss") == "train/loss"
    assert model._metric_key("val", "acc") == "val/acc"
    assert model._metric_key("test", "consistency") == "test/consistency"
