import json

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from src.data.coco_karpathy_datamodule import (
    CocoKarpathyRetrievalDataModule,
    CocoKarpathyRetrievalDataset,
)
from src.evaluation.coco_karpathy import compute_coco_karpathy_metrics


def test_coco_karpathy_dataset_loads_split_and_mappings(tmp_path):
    image_dir = tmp_path / "images" / "val2014"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color="white").save(image_dir / "COCO_val2014_000000000001.jpg")

    annotation_file = tmp_path / "coco_karpathy_test.json"
    annotation_file.write_text(
        json.dumps(
            [
                {
                    "image": "val2014/COCO_val2014_000000000001.jpg",
                    "caption": ["A white square.", "A small bright image."],
                }
            ]
        ),
        encoding="utf-8",
    )

    dataset = CocoKarpathyRetrievalDataset(
        annotation_file=str(annotation_file),
        image_root=str(tmp_path / "images"),
    )

    assert len(dataset) == 1
    assert dataset.text == ["a white square", "a small bright image"]
    assert dataset.img2txt == {0: [0, 1]}
    assert dataset.txt2img == {0: 0, 1: 0}
    assert dataset[0]["filename"] == "val2014/COCO_val2014_000000000001.jpg"


def test_coco_karpathy_dataset_groups_flat_caption_rows_by_image(tmp_path):
    image_dir = tmp_path / "images" / "train2014"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color="white").save(image_dir / "COCO_train2014_000000000001.jpg")
    Image.new("RGB", (8, 8), color="black").save(image_dir / "COCO_train2014_000000000002.jpg")

    annotation_file = tmp_path / "coco_karpathy_train.json"
    annotation_file.write_text(
        json.dumps(
            [
                {
                    "image": "train2014/COCO_train2014_000000000001.jpg",
                    "caption": "A white square.",
                },
                {
                    "image": "train2014/COCO_train2014_000000000001.jpg",
                    "caption": "A bright sample.",
                },
                {
                    "image": "train2014/COCO_train2014_000000000002.jpg",
                    "caption": "A black square.",
                },
            ]
        ),
        encoding="utf-8",
    )

    dataset = CocoKarpathyRetrievalDataset(
        annotation_file=str(annotation_file),
        image_root=str(tmp_path / "images"),
        split="train",
    )

    assert len(dataset) == 2
    assert dataset.img2txt == {0: [0, 1], 1: [2]}
    assert dataset.txt2img == {0: 0, 1: 0, 2: 1}
    assert dataset.text == ["a white square", "a bright sample", "a black square"]
    sample = dataset[0]
    assert sample["filename"] == "train2014/COCO_train2014_000000000001.jpg"
    assert sample["caption"] in {"a white square", "a bright sample"}


def test_coco_karpathy_dataset_reports_missing_images(tmp_path):
    annotation_file = tmp_path / "coco_karpathy_test.json"
    annotation_file.write_text(
        json.dumps([{"image": "val2014/missing.jpg", "caption": ["missing image"]}]),
        encoding="utf-8",
    )
    dataset = CocoKarpathyRetrievalDataset(
        annotation_file=str(annotation_file),
        image_root=str(tmp_path / "images"),
    )

    with pytest.raises(FileNotFoundError, match="Missing COCO Karpathy image"):
        dataset[0]


def test_coco_karpathy_metrics_recall_and_modality_gap():
    image_features = torch.eye(3)
    text_features = torch.eye(3).repeat_interleave(2, dim=0)
    img2txt = {0: [0, 1], 1: [2, 3], 2: [4, 5]}
    txt2img = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}

    metrics = compute_coco_karpathy_metrics(
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


def test_coco_karpathy_datamodule_supports_train_val_test_splits(tmp_path):
    image_dir = tmp_path / "images" / "val2014"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color="white").save(image_dir / "sample.jpg")
    annotation_dir = tmp_path / "karpathy_splits"
    annotation_dir.mkdir()
    row = {"image": "val2014/sample.jpg", "caption": ["A sample image.", "Another caption."]}
    for split in ("train", "val", "test"):
        (annotation_dir / f"coco_karpathy_{split}.json").write_text(
            json.dumps([row]),
            encoding="utf-8",
        )
    datamodule = CocoKarpathyRetrievalDataModule(
        annotation_dir=str(annotation_dir),
        image_root=str(tmp_path / "images"),
        eval_split="val",
    )

    datamodule.setup("fit")
    assert datamodule.train_set.split == "train"
    assert "caption" in datamodule.train_set[0]
    assert datamodule.val_set.split == "val"

    datamodule.setup("test")
    assert datamodule.test_set.split == "test"
