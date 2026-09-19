import json

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from src.data.flickr30k_datamodule import (
    Flickr30KRetrievalDataModule,
    Flickr30KRetrievalDataset,
)
from src.evaluation.flickr30k import compute_flickr30k_metrics
from src.evaluation.retrieval import RetrievalAdapterHead, RetrievalAdapterTuneEvaluator


def test_flickr30k_test_dataset_loads_five_caption_retrieval_split(tmp_path):
    image_root = tmp_path / "Images"
    image_root.mkdir()
    Image.new("RGB", (8, 8), color="white").save(image_root / "100.jpg")
    annotation_file = tmp_path / "flickr30k_test.json"
    annotation_file.write_text(
        json.dumps(
            [
                {
                    "image": "flickr30k-images/100.jpg",
                    "caption": ["A white image.", "A small white square."],
                }
            ]
        ),
        encoding="utf-8",
    )

    dataset = Flickr30KRetrievalDataset(
        annotation_file=str(annotation_file),
        image_root=str(image_root),
        split="test",
    )

    assert len(dataset) == 1
    assert dataset.text == ["a white image", "a small white square"]
    assert dataset.img2txt == {0: [0, 1]}
    assert dataset.txt2img == {0: 0, 1: 0}
    assert dataset[0]["filename"] == "flickr30k-images/100.jpg"


def test_flickr30k_train_dataset_returns_caption_pairs(tmp_path):
    image_root = tmp_path / "Images"
    image_root.mkdir()
    Image.new("RGB", (8, 8), color="white").save(image_root / "101.jpg")
    annotation_file = tmp_path / "flickr30k_train.json"
    annotation_file.write_text(
        json.dumps(
            [
                {
                    "image": "flickr30k-images/101.jpg",
                    "caption": "Two people walk outside.",
                    "image_id": 0,
                }
            ]
        ),
        encoding="utf-8",
    )

    dataset = Flickr30KRetrievalDataset(
        annotation_file=str(annotation_file),
        image_root=str(image_root),
        split="train",
    )

    assert dataset[0]["caption"] == "two people walk outside"


def test_flickr30k_dataset_reports_missing_images(tmp_path):
    annotation_file = tmp_path / "flickr30k_test.json"
    annotation_file.write_text(
        json.dumps([{"image": "flickr30k-images/missing.jpg", "caption": ["missing image"]}]),
        encoding="utf-8",
    )
    dataset = Flickr30KRetrievalDataset(
        annotation_file=str(annotation_file),
        image_root=str(tmp_path / "Images"),
        split="test",
    )

    with pytest.raises(FileNotFoundError, match="Missing Flickr30K image"):
        dataset[0]


def test_flickr30k_metrics_recall_and_modality_gap():
    image_features = torch.eye(3)
    text_features = torch.eye(3).repeat_interleave(2, dim=0)
    img2txt = {0: [0, 1], 1: [2, 3], 2: [4, 5]}
    txt2img = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}

    metrics = compute_flickr30k_metrics(
        image_features=image_features,
        text_features=text_features,
        txt2img=txt2img,
        img2txt=img2txt,
    )

    assert metrics["i2t_r1"] == 1.0
    assert metrics["t2i_r1"] == 1.0
    assert metrics["i2t_r5"] == 1.0
    assert metrics["t2i_r10"] == 1.0
    assert metrics["modality_gap"] == pytest.approx(0.0)


def test_flickr30k_datamodule_uses_explicit_eval_split(tmp_path):
    annotation_dir = tmp_path / "annotations"
    image_root = tmp_path / "Images"
    annotation_dir.mkdir()
    image_root.mkdir()
    for image_id in ("val", "test"):
        Image.new("RGB", (8, 8), color="white").save(image_root / f"{image_id}.jpg")
        (annotation_dir / f"flickr30k_{image_id}.json").write_text(
            json.dumps(
                [
                    {
                        "image": f"{image_id}.jpg",
                        "caption": [f"{image_id} caption"],
                    }
                ]
            ),
            encoding="utf-8",
        )
    (annotation_dir / "flickr30k_train.json").write_text(
        json.dumps([{"image": "val.jpg", "caption": ["train caption"]}]),
        encoding="utf-8",
    )
    datamodule = Flickr30KRetrievalDataModule(
        annotation_dir=str(annotation_dir),
        image_root=str(image_root),
        eval_split="val",
    )

    datamodule.setup("validate")

    assert datamodule.val_set.split == "val"
    assert datamodule.val_set.annotation_file.name == "flickr30k_val.json"


def test_retrieval_adapter_head_outputs_l2_normalized_projection():
    head = RetrievalAdapterHead(input_dim=4, projection_dim=3)
    output = head(torch.randn(5, 4))

    assert output.shape == (5, 3)
    assert torch.allclose(torch.linalg.vector_norm(output, dim=-1), torch.ones(5), atol=1e-6)


def test_retrieval_residual_adapter_starts_as_noop_when_dimensions_match():
    head = RetrievalAdapterHead(input_dim=4, projection_dim=4, residual=True)
    features = torch.randn(5, 4)

    output = head(features)

    assert torch.allclose(output, torch.nn.functional.normalize(features, dim=-1), atol=1e-6)
    assert head.residual_scale.item() == pytest.approx(0.0)


class _TinyRetrievalAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)

    def encode_images(self, images, normalize=True):
        del normalize
        return images

    def encode_texts(self, texts, normalize=True):
        del normalize
        return torch.eye(4)[: len(list(texts))]


def test_retrieval_adapter_tuning_freezes_backbone_and_trains_adapters():
    adapter = _TinyRetrievalAdapter()
    evaluator = RetrievalAdapterTuneEvaluator(
        ckpt_path="",
        metric="flickr30k",
        mode="retrieval_adapter_tune",
        feature_dim=4,
        projection_dim=3,
        adapter=adapter,
    )

    assert all(not param.requires_grad for param in evaluator.adapter.parameters())
    assert all(param.requires_grad for param in evaluator.image_adapter.parameters())
    assert all(param.requires_grad for param in evaluator.text_adapter.parameters())
    assert evaluator.logit_scale.requires_grad is True
    assert evaluator.logit_scale.item() == pytest.approx(2.6592)
    assert evaluator.image_adapter.use_residual is False
