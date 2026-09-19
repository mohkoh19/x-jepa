from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

from src.models.base import compute_optimizer_step_schedule


def _trainer(num_training_batches=100, accumulate_grad_batches=1, max_epochs=2):
    return SimpleNamespace(
        num_training_batches=num_training_batches,
        accumulate_grad_batches=accumulate_grad_batches,
        max_epochs=max_epochs,
    )


def test_optimizer_steps_per_epoch_matches_batches_without_accumulation():
    schedule = compute_optimizer_step_schedule(
        _trainer(num_training_batches=100, accumulate_grad_batches=1),
        ipe_scale=1.0,
        warmup=4,
    )

    assert schedule.num_training_batches == 100
    assert schedule.accumulate_grad_batches == 1
    assert schedule.optimizer_steps_per_epoch == 100
    assert schedule.total_optimizer_steps == 200


def test_optimizer_steps_per_epoch_uses_gradient_accumulation():
    schedule = compute_optimizer_step_schedule(
        _trainer(num_training_batches=100, accumulate_grad_batches=4),
        ipe_scale=1.0,
        warmup=4,
    )

    assert schedule.num_training_batches == 100
    assert schedule.accumulate_grad_batches == 4
    assert schedule.optimizer_steps_per_epoch == 25
    assert schedule.total_optimizer_steps == 50


def test_warmup_steps_use_optimizer_steps_not_raw_batches():
    schedule = compute_optimizer_step_schedule(
        _trainer(num_training_batches=100, accumulate_grad_batches=4, max_epochs=40),
        ipe_scale=1.0,
        warmup=4,
    )

    assert schedule.optimizer_steps_per_epoch == 25
    assert schedule.total_optimizer_steps == 1000
    assert schedule.warmup_optimizer_steps == 100
