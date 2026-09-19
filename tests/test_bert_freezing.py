from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("lightning")
pytest.importorskip("transformers")

import torch.nn as nn
from transformers import BertConfig, BertModel

from src.models.base import BaseModule


def _bert(hidden_size: int = 8, layers: int = 4, vocab_size: int = 128) -> BertModel:
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=layers,
        num_attention_heads=2,
        max_position_embeddings=16,
    )
    return BertModel(config)


class DummyBertModule(BaseModule):
    def __init__(self, text_encoder: nn.Module):
        super().__init__()
        self.text_encoder = text_encoder


def _all_params_require_grad(module: nn.Module) -> bool:
    return all(param.requires_grad for param in module.parameters())


def _no_params_require_grad(module: nn.Module) -> bool:
    return not any(param.requires_grad for param in module.parameters())


def test_unfreeze_last_n_negative_unfreezes_all_params_and_sets_train_mode():
    model = DummyBertModule(_bert(layers=2))

    model.unfreeze_bert_layers(unfreeze_last_n=-1)

    assert _all_params_require_grad(model.text_encoder)

    model.train(True)
    assert model.text_encoder.training is True


def test_unfreeze_last_n_zero_freezes_all_params_and_keeps_eval_mode_even_in_train():
    model = DummyBertModule(_bert(layers=2))

    model.unfreeze_bert_layers(unfreeze_last_n=0)

    assert _no_params_require_grad(model.text_encoder)
    assert model.text_encoder.training is False

    # BaseModule.train() should keep a fully frozen encoder in eval mode.
    model.train(True)
    assert model.text_encoder.training is False


def test_unfreeze_last_n_one_unfreezes_only_last_encoder_layer_and_pooler():
    num_layers = 4
    model = DummyBertModule(_bert(layers=num_layers))

    model.unfreeze_bert_layers(unfreeze_last_n=1)

    # Embeddings remain frozen.
    assert all(not p.requires_grad for p in model.text_encoder.embeddings.parameters())

    # Only the last encoder layer is trainable.
    layers = model.text_encoder.encoder.layer
    for idx, layer in enumerate(layers):
        layer_params = list(layer.parameters())
        assert layer_params, "Expected encoder layer to have parameters"
        if idx == num_layers - 1:
            assert all(p.requires_grad for p in layer_params)
        else:
            assert all(not p.requires_grad for p in layer_params)

    # Historical behaviour: pooler becomes trainable when unfreezing encoder layers.
    assert model.text_encoder.pooler is not None
    assert any(p.requires_grad for p in model.text_encoder.pooler.parameters())

    model.train(True)
    assert model.text_encoder.training is True


def test_unfreeze_last_n_clamps_to_num_layers_without_unfreezing_embeddings():
    num_layers = 3
    model = DummyBertModule(_bert(layers=num_layers))

    model.unfreeze_bert_layers(unfreeze_last_n=num_layers + 5)

    # All encoder layers should be trainable, but embeddings stay frozen.
    assert all(not p.requires_grad for p in model.text_encoder.embeddings.parameters())

    for layer in model.text_encoder.encoder.layer:
        assert all(p.requires_grad for p in layer.parameters())

    model.train(True)
    assert model.text_encoder.training is True
