from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.build_vl4m_webdataset import build_webdataset


def _require_module(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        pytest.skip(f"{name} is not installed")


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path, format="JPEG")


def _make_manifest(root: Path) -> Path:
    rows = [
        {
            "image_path": "cc3m/images/000000000.jpg",
            "caption": "A River! With Punctuation.",
        },
        {
            "image_path": "sbu/dataset/000000001.jpg",
            "caption": "Second caption",
        },
        {
            "image_path": "vg/image/000000002.jpg",
            "caption": "Third caption",
        },
        {
            "image_path": "coco/images/000000003.jpg",
            "caption": "Fourth caption",
        },
    ]
    for idx, row in enumerate(rows):
        _write_image(root / row["image_path"], (idx * 40, 80, 120))

    manifest = root / "4M_cleaned.json"
    manifest.write_text(json.dumps(rows), encoding="utf-8")
    return manifest


def test_vl4m_webdataset_builder_and_datamodule_preserve_batch_contract(tmp_path: Path):
    _require_module("torch")
    _require_module("lightning")
    _require_module("webdataset")

    import numpy as np
    import torch

    from src.data.VL4M_datamodule import VL4MWebDatasetDatamodule

    root = tmp_path / "data"
    manifest = _make_manifest(root)
    output_dir = tmp_path / "wds"

    report = build_webdataset(
        argparse.Namespace(
            input_json=str(manifest),
            root_dir=str(root),
            output_dir=str(output_dir),
            pattern="vl4m-%06d.tar",
            maxcount=2,
            maxsize_gb=1.0,
            limit=None,
            progress_every=10_000,
            verify_images=True,
            overwrite=False,
        )
    )

    assert report.samples == 4
    assert len(report.source_sha256) == 64
    assert report.expected_samples is None
    assert len(report.shards) == 2
    assert (output_dir / "dataset.json").exists()

    def transform(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.resize((224, 224)), dtype=np.float32)
        array = array.transpose(2, 0, 1) / 255.0
        return torch.from_numpy(array)

    datamodule = VL4MWebDatasetDatamodule(
        train_shards=str(output_dir / "vl4m-{000000..000001}.tar"),
        metadata_path=str(output_dir / "dataset.json"),
        val_frac=0.0,
        batch_size=2,
        train_transform=transform,
        val_transform=transform,
        num_workers=0,
        pin_memory=False,
        shuffle_buffer=1,
        shard_shuffle=1,
        seed=0,
    )
    datamodule.setup("fit")

    loader = datamodule.train_dataloader()
    assert len(loader) == 2
    batch = next(iter(loader))
    images, captions, labels, indices, sources = batch
    caption_values, paired_caption_values = captions

    assert tuple(images.shape) == (2, 3, 224, 224)
    assert labels.tolist() == [-1, -1]
    assert len(caption_values) == 2
    assert all(caption == caption.lower() for caption in caption_values)
    assert all("." not in caption and "!" not in caption for caption in caption_values)
    assert paired_caption_values == ("", "")
    assert all(int(index) in {0, 1, 2, 3} for index in indices)
    assert set(sources).issubset({"CC3M", "SBU", "VG", "COCO"})


def test_vl4m_webdataset_validation_loader_reads_from_shards(tmp_path: Path):
    _require_module("torch")
    _require_module("lightning")
    _require_module("webdataset")

    import numpy as np
    import torch

    from src.data.VL4M_datamodule import VL4MWebDatasetDatamodule

    root = tmp_path / "data"
    manifest = _make_manifest(root)
    output_dir = tmp_path / "wds"

    build_webdataset(
        argparse.Namespace(
            input_json=str(manifest),
            root_dir=str(root),
            output_dir=str(output_dir),
            pattern="vl4m-%06d.tar",
            maxcount=2,
            maxsize_gb=1.0,
            limit=None,
            progress_every=10_000,
            verify_images=True,
            overwrite=False,
        )
    )

    def transform(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.resize((224, 224)), dtype=np.float32)
        array = array.transpose(2, 0, 1) / 255.0
        return torch.from_numpy(array)

    datamodule = VL4MWebDatasetDatamodule(
        train_shards=str(output_dir / "vl4m-{000000..000001}.tar"),
        metadata_path=str(output_dir / "dataset.json"),
        val_frac=1.0,
        batch_size=2,
        train_transform=transform,
        val_transform=transform,
        num_workers=0,
        pin_memory=False,
        shuffle_buffer=1,
        shard_shuffle=1,
        seed=0,
    )
    datamodule.setup("fit")

    loader = datamodule.val_dataloader()
    assert loader is not None
    assert len(loader) == 2
    batch = next(iter(loader))
    images, captions, labels, indices, sources = batch
    caption_values, paired_caption_values = captions

    assert tuple(images.shape) == (2, 3, 224, 224)
    assert labels.tolist() == [-1, -1]
    assert len(caption_values) == 2
    assert paired_caption_values == ("", "")
    assert all(int(index) in {0, 1, 2, 3} for index in indices)
    assert set(sources).issubset({"CC3M", "SBU", "VG", "COCO"})


def test_vl4m_webdataset_split_predicate_is_deterministic_and_complementary():
    from src.data.VL4M_datamodule import VL4MWebDatasetSplitPredicate

    samples = [{"__key__": f"{idx:09d}", "__url__": "vl4m-000000.tar"} for idx in range(100)]
    val_predicate = VL4MWebDatasetSplitPredicate(split="val", val_frac=0.25, seed=13)
    train_predicate = VL4MWebDatasetSplitPredicate(split="train", val_frac=0.25, seed=13)

    val_keys = {sample["__key__"] for sample in samples if val_predicate(sample)}
    train_keys = {sample["__key__"] for sample in samples if train_predicate(sample)}

    assert 0 < len(val_keys) < len(samples)
    assert val_keys.isdisjoint(train_keys)
    assert val_keys | train_keys == {sample["__key__"] for sample in samples}


def test_vl4m_webdataset_rejects_invalid_val_frac(tmp_path: Path):
    from src.data.VL4M_datamodule import VL4MWebDatasetDatamodule

    with pytest.raises(ValueError, match="val_frac"):
        VL4MWebDatasetDatamodule(
            train_shards=str(tmp_path / "vl4m-{000000..000001}.tar"),
            dataset_size=4,
            val_frac=1.5,
        )
