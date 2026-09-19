import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("datasets")

from src.data import svo_probes_datamodule
from src.data.svo_probes_datamodule import SVOProbesDataModule, SVOProbesDataset
from src.evaluation.svo_probes import (
    compute_svo_probes_metrics,
    evaluate_svo_probes_task,
    score_svo_probes_examples,
)


class _TinyHFDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def _rows():
    return [
        {
            "sentence": "a person rides a horse",
            "pos_triplet": "person ride horse",
            "neg_triplet": "dog ride horse",
            "pos_url": "https://example.test/1.jpg",
            "neg_url": "https://example.test/2.jpg",
            "pos_image_id": 1,
            "neg_image_id": 2,
            "subj_neg": True,
            "verb_neg": False,
            "obj_neg": False,
        },
        {
            "sentence": "a dog catches a ball",
            "pos_triplet": "dog catch ball",
            "neg_triplet": "dog eat ball",
            "pos_url": "https://example.test/3.jpg",
            "neg_url": "https://example.test/4.jpg",
            "pos_image_id": 3,
            "neg_image_id": 4,
            "subj_neg": False,
            "verb_neg": True,
            "obj_neg": False,
        },
    ]


def _write_images(image_root):
    image_root.mkdir(parents=True)
    for image_id, color in [(1, "white"), (2, "black"), (3, "red"), (4, "blue")]:
        Image.new("RGB", (8, 8), color=color).save(image_root / f"{image_id:06d}.jpg")


def test_svo_probes_dataset_wraps_huggingface_rows(monkeypatch, tmp_path):
    image_root = tmp_path / "images"
    _write_images(image_root)
    monkeypatch.setattr(
        svo_probes_datamodule,
        "load_dataset",
        lambda dataset_name, split, cache_dir=None: _TinyHFDataset(_rows()),
    )

    dataset = SVOProbesDataset(
        image_root=str(image_root),
        transform=lambda image: torch.zeros(3, 4, 4),
    )

    sample = dataset[0]
    assert sample["pos_image"].shape == (3, 4, 4)
    assert sample["neg_image"].shape == (3, 4, 4)
    assert sample["sentence"] == "a person rides a horse"
    assert sample["negative_type"] == "subject"
    assert sample["pos_image_id"] == 1
    assert sample["neg_image_id"] == 2


def test_svo_probes_datamodule_batches_expected_fields(monkeypatch, tmp_path):
    image_root = tmp_path / "images"
    _write_images(image_root)
    monkeypatch.setattr(
        svo_probes_datamodule,
        "load_dataset",
        lambda dataset_name, split, cache_dir=None: _TinyHFDataset(_rows()),
    )
    datamodule = SVOProbesDataModule(
        image_root=str(image_root),
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        transform=lambda image: torch.ones(3, 4, 4),
    )

    datamodule.setup("validate")
    batch = next(iter(datamodule.val_dataloader()))

    assert batch["pos_image"].shape == (2, 3, 4, 4)
    assert batch["neg_image"].shape == (2, 3, 4, 4)
    assert list(batch["sentence"]) == [
        "a person rides a horse",
        "a dog catches a ball",
    ]
    assert list(batch["negative_type"]) == ["subject", "verb"]
    assert batch["pos_image_id"].tolist() == [1, 3]
    assert batch["neg_image_id"].tolist() == [2, 4]


def test_compute_svo_probes_metrics_overall_and_by_negative_type():
    metrics = compute_svo_probes_metrics(
        [
            {"negative_type": "subject", "correct": True, "margin": 0.5},
            {"negative_type": "verb", "correct": False, "margin": -0.2},
            {"negative_type": "object", "correct": True, "margin": 0.1},
        ]
    )

    assert metrics["acc"] == pytest.approx(2 / 3)
    assert metrics["margin_mean"] == pytest.approx((0.5 - 0.2 + 0.1) / 3)
    assert metrics["n_examples"] == 3.0
    assert metrics["subject_acc"] == pytest.approx(1.0)
    assert metrics["verb_acc"] == pytest.approx(0.0)
    assert metrics["object_margin_mean"] == pytest.approx(0.1)


class _ToySVODataModule:
    def setup(self, stage=None):
        self.stage = stage

    def val_dataloader(self):
        return [
            {
                "pos_image": torch.eye(2),
                "neg_image": torch.flip(torch.eye(2), dims=[0]),
                "sentence": ["subject sentence", "verb sentence"],
                "negative_type": ["subject", "verb"],
            }
        ]


class _ToySVOAdapter:
    def encode_images(self, images, normalize=True):
        del normalize
        return images

    def encode_texts(self, texts, normalize=True):
        del normalize
        lookup = {
            "subject sentence": torch.tensor([1.0, 0.0]),
            "verb sentence": torch.tensor([0.0, 1.0]),
        }
        return torch.stack([lookup[text] for text in texts])


def test_svo_probes_task_returns_accuracy_and_margin():
    metrics = evaluate_svo_probes_task(_ToySVOAdapter(), _ToySVODataModule())

    assert metrics["acc"] == pytest.approx(1.0)
    assert metrics["margin_mean"] == pytest.approx(1.0)
    assert metrics["n_examples"] == 2.0
    assert set(metrics) == {"acc", "margin_mean", "n_examples"}


def test_svo_probes_per_example_scores_preserve_native_group():
    rows = score_svo_probes_examples(_ToySVOAdapter(), _ToySVODataModule(), split="train")

    assert rows[0]["dataset"] == "svo"
    assert rows[0]["negative_type_native"] == "subject"
    assert rows[0]["negative_type_group"] == "subject"
    assert rows[0]["sentence"] == "subject sentence"
    assert rows[0]["margin"] == pytest.approx(1.0)
    assert rows[0]["correct"] is True
