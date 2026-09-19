import json

import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from src.data.nlvr2_datamodule import NLVR2Dataset
from src.evaluation.nlvr2 import (
    NLVR2Evaluator,
    NLVR2InteractionHead,
    NLVR2TokenInteractionHead,
    compute_nlvr2_metrics,
)


def _write_image(path, color="white"):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


def test_nlvr2_dataset_loads_explicit_image_paths_and_bool_labels(tmp_path):
    image_dir = tmp_path / "images"
    _write_image(image_dir / "left.png")
    _write_image(image_dir / "right.png", color="black")
    annotation_file = tmp_path / "train.jsonl"
    annotation_file.write_text(
        json.dumps(
            {
                "image_left": "left.png",
                "image_right": "right.png",
                "sentence": "The two images differ.",
                "label": True,
                "identifier": "train-1-0-0",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = NLVR2Dataset(annotation_file=str(annotation_file), image_dir=str(image_dir))

    item = dataset[0]
    assert item["sentence"] == "The two images differ."
    assert item["label"] == 1
    assert item["identifier"] == "train-1-0-0"


def test_nlvr2_dataset_infers_official_image_paths_from_identifier(tmp_path):
    image_dir = tmp_path / "images"
    _write_image(image_dir / "train-42-3-img0.png")
    _write_image(image_dir / "train-42-3-img1.png", color="black")
    annotation_file = tmp_path / "train.json"
    annotation_file.write_text(
        json.dumps(
            [
                {
                    "identifier": "train-42-3-0",
                    "sentence": "A sentence over an image pair.",
                    "label": "False",
                }
            ]
        ),
        encoding="utf-8",
    )

    dataset = NLVR2Dataset(annotation_file=str(annotation_file), image_dir=str(image_dir))

    assert dataset[0]["label"] == 0
    assert dataset[0]["identifier"] == "train-42-3-0"


@pytest.mark.parametrize(
    "annotation_name,identifier,left_rel,right_rel,directory",
    [
        (
            "train.json",
            "train-42-3-0",
            "train_img/images/train/11/train-42-3-img0.png",
            "train_img/images/train/11/train-42-3-img1.png",
            11,
        ),
        (
            "dev.json",
            "dev-7-0-0",
            "dev_img/dev/dev-7-0-img0.png",
            "dev_img/dev/dev-7-0-img1.png",
            None,
        ),
        (
            "test1.json",
            "test1-9-1-0",
            "test1_img/test1/test1-9-1-img0.png",
            "test1_img/test1/test1-9-1-img1.png",
            None,
        ),
    ],
)
def test_nlvr2_dataset_prefers_split_specific_roots(
    tmp_path, annotation_name, identifier, left_rel, right_rel, directory
):
    image_root = tmp_path
    _write_image(image_root / left_rel)
    _write_image(image_root / right_rel, color="black")

    row = {
        "identifier": identifier,
        "sentence": "A sentence over an image pair.",
        "label": "true",
    }
    if directory is not None:
        row["directory"] = directory

    annotation_file = tmp_path / annotation_name
    annotation_file.write_text(json.dumps([row]), encoding="utf-8")

    dataset = NLVR2Dataset(annotation_file=str(annotation_file), image_dir=str(image_root))

    item = dataset[0]
    assert item["identifier"] == identifier
    assert item["label"] == 1


def test_nlvr2_dataset_reports_missing_images(tmp_path):
    annotation_file = tmp_path / "dev.jsonl"
    annotation_file.write_text(
        json.dumps(
            {
                "image_left": "missing-left.png",
                "image_right": "missing-right.png",
                "sentence": "Missing images.",
                "label": "true",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = NLVR2Dataset(
        annotation_file=str(annotation_file), image_dir=str(tmp_path / "images")
    )

    with pytest.raises(FileNotFoundError, match="Missing NLVR2 image"):
        dataset[0]


def test_nlvr2_metrics_match_official_accuracy_and_consistency_grouping():
    metrics = compute_nlvr2_metrics(
        predictions=[1, 1, 1, 0],
        labels=[1, 0, 1, 0],
        identifiers=["dev-10-0-0", "dev-10-1-0", "dev-11-0-0", "dev-12-0-0"],
        sentences=[
            "The paired examples share one sentence.",
            "The paired examples share one sentence.",
            "Another sentence.",
            "A third sentence.",
        ],
    )

    assert metrics["acc"] == pytest.approx(0.75)
    assert metrics["consistency"] == pytest.approx(2 / 3)


def test_nlvr2_interaction_head_uses_frozen_probe_feature_product():
    head = NLVR2InteractionHead(feature_dim=2, hidden_dim=4, dropout=0.0)
    logits = head(torch.ones(3, 2), torch.zeros(3, 2), torch.ones(3, 2))

    assert head.net[0].normalized_shape == (16,)
    assert logits.shape == (3, 2)


def test_nlvr2_token_interaction_head_fuses_image_and_text_tokens():
    head = NLVR2TokenInteractionHead(
        feature_dim=4,
        fusion_dim=8,
        num_layers=1,
        num_heads=2,
        classifier_hidden_dim=12,
        image_token_pool=2,
        dropout=0.0,
    )

    logits = head(
        left_tokens=torch.ones(2, 16, 4),
        right_tokens=torch.zeros(2, 16, 4),
        text_tokens=torch.randn(2, 5, 4),
        text_mask=torch.tensor([[True, True, True, False, False], [True] * 5]),
    )

    assert logits.shape == (2, 2)


class _TinyNLVR2Adapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)


def test_nlvr2_default_global_projected_mode_freezes_backbone_and_trains_classifier():
    adapter = _TinyNLVR2Adapter()
    evaluator = NLVR2Evaluator(ckpt_path="", adapter=adapter, feature_dim=2)

    assert evaluator.hparams.mode == "frozen_probe"
    assert evaluator.hparams.freeze_backbone is True
    assert all(not param.requires_grad for param in evaluator.adapter.parameters())
    assert all(param.requires_grad for param in evaluator.classifier.parameters())


class _TinyTokenInteractionAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)

    def encode_image_sequence(self, images, normalize=False, project=False):
        del normalize, project
        batch_size = images.shape[0]
        return images.new_ones(batch_size, 4, 2)

    def encode_text_sequence(self, texts, normalize=False, project=False):
        del normalize, project
        batch_size = len(texts)
        tokens = torch.ones(batch_size, 3, 2)
        mask = torch.ones(batch_size, 3, dtype=torch.bool)
        return tokens, mask


def test_nlvr2_token_interaction_mode_freezes_backbone_and_trains_probe():
    adapter = _TinyTokenInteractionAdapter()
    evaluator = NLVR2Evaluator(
        ckpt_path="",
        adapter=adapter,
        mode="token_interaction",
        feature_dim=2,
        token_interaction_dim=8,
        token_interaction_layers=1,
        token_interaction_heads=2,
        token_interaction_classifier_hidden_dim=12,
        token_interaction_image_pool=2,
    )

    batch = {
        "image_left": torch.zeros(2, 2),
        "image_right": torch.zeros(2, 2),
        "sentence": ["A caption.", "Another caption."],
    }

    assert evaluator.probe_mode == "token_interaction"
    assert evaluator.hparams.feature_projection == "raw"
    assert all(not param.requires_grad for param in evaluator.adapter.parameters())
    assert all(param.requires_grad for param in evaluator.classifier.parameters())
    assert evaluator._logits(batch).shape == (2, 2)


def test_nlvr2_removed_token_interaction_alias_fails_clearly():
    with pytest.raises(ValueError, match="Unsupported NLVR2 evaluation mode"):
        NLVR2Evaluator(
            ckpt_path="",
            adapter=_TinyTokenInteractionAdapter(),
            mode="token-interaction",
            feature_dim=2,
        )


class _TinyNativeFusedAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.calls = []

    def encode_image_text_features(
        self,
        images,
        texts,
        feature_source="unimodal_mean",
        normalize=True,
    ):
        del texts, normalize
        self.calls.append(feature_source)
        return images
