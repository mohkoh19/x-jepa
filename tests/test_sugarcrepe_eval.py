import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from src.data.sugarcrepe_datamodule import SugarCrepePPDataset
from src.evaluation.sugarcrepe import (
    compute_sugarcrepe_metrics,
    evaluate_sugarcrepe_task,
    score_sugarcrepe_examples,
)


def _write_jsonl(path, rows):
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_sugarcrepe_dataset_parses_jsonl_and_normalizes_category(tmp_path):
    annotation_dir = tmp_path / "scpp"
    image_dir = tmp_path / "images"
    annotation_dir.mkdir()
    image_dir.mkdir()
    Image.new("RGB", (8, 8), color="white").save(image_dir / "000000000001.jpg")
    _write_jsonl(
        annotation_dir / "swap_atribute_train.jsonl",
        [
            (
                '{"id": 7, "filename": "000000000001.jpg", "caption": "a red cup", '
                '"caption2": "a cup that is red", "negative_caption": "a blue cup", '
                '"category": "swap_atribute"}'
            )
        ],
    )

    dataset = SugarCrepePPDataset(
        annotation_dir=str(annotation_dir),
        image_dir=str(image_dir),
        files=["swap_atribute_train.jsonl"],
    )

    sample = dataset[0]
    assert sample["category"] == "swap_attribute"
    assert sample["caption"] == "a red cup"
    assert sample["caption2"] == "a cup that is red"
    assert sample["negative_caption"] == "a blue cup"


def test_sugarcrepe_dataset_raises_for_missing_image(tmp_path):
    annotation_dir = tmp_path / "scpp"
    image_dir = tmp_path / "images"
    annotation_dir.mkdir()
    image_dir.mkdir()
    _write_jsonl(
        annotation_dir / "replace_object_train.jsonl",
        [
            (
                '{"id": 1, "filename": "missing.jpg", "caption": "a cat", '
                '"caption2": "one cat", "negative_caption": "a dog", '
                '"category": "replace_object"}'
            )
        ],
    )
    dataset = SugarCrepePPDataset(
        annotation_dir=str(annotation_dir),
        image_dir=str(image_dir),
        files=["replace_object_train.jsonl"],
    )

    with pytest.raises(FileNotFoundError, match="missing.jpg"):
        dataset[0]


def test_compute_sugarcrepe_metrics_overall_and_by_category():
    metrics = compute_sugarcrepe_metrics(
        [
            {"category": "replace_object", "itt": True, "p1": True, "p2": True},
            {"category": "replace_object", "itt": False, "p1": True, "p2": False},
            {"category": "swap_attribute", "itt": False, "p1": False, "p2": True},
        ]
    )

    assert metrics["itt_acc"] == pytest.approx(1 / 3)
    assert metrics["p1_acc"] == pytest.approx(2 / 3)
    assert metrics["p2_acc"] == pytest.approx(2 / 3)
    assert metrics["replace_object_itt_acc"] == pytest.approx(0.5)
    assert metrics["swap_attribute_p2_acc"] == pytest.approx(1.0)


def test_sugarcrepe_metric_inputs_accept_tensor_bools():
    metrics = compute_sugarcrepe_metrics(
        [{"category": "replace_relation", "itt": torch.tensor(True), "p1": True, "p2": True}]
    )

    assert metrics["replace_relation_itt_acc"] == 1.0


class _ToySugarDataModule:
    def setup(self, stage=None):
        self.stage = stage

    def val_dataloader(self):
        return [
            {
                "image": torch.eye(2),
                "caption": ["p1", "p2"],
                "caption2": ["p1b", "p2b"],
                "negative_caption": ["n1", "n2"],
                "category": ["replace_object", "replace_object"],
            }
        ]


class _ToySugarAdapter:
    def encode_images(self, images, normalize=True):
        del normalize
        return images

    def encode_texts(self, texts, normalize=True):
        del normalize
        lookup = {
            "p1": torch.tensor([1.0, 0.0]),
            "p2": torch.tensor([0.0, 1.0]),
            "p1b": torch.tensor([1.0, 0.0]),
            "p2b": torch.tensor([1.0, 0.0]),
            "n1": torch.tensor([0.0, 1.0]),
            "n2": torch.tensor([1.0, 0.0]),
        }
        return torch.stack([lookup[text] for text in texts])


def test_sugarcrepe_task_returns_itt_acc():
    metrics = evaluate_sugarcrepe_task(_ToySugarAdapter(), _ToySugarDataModule())

    assert metrics["itt_acc"] == pytest.approx(0.5)
    assert set(metrics) == {"itt_acc", "p1_acc", "p2_acc"}


def test_sugarcrepe_per_example_scores_export_required_fields():
    rows = score_sugarcrepe_examples(_ToySugarAdapter(), _ToySugarDataModule())

    assert rows[0]["dataset"] == "sugarcrepe_pp"
    assert rows[0]["positive_caption_1"] == "p1"
    assert rows[0]["positive_caption_2"] == "p1b"
    assert rows[0]["negative_caption"] == "n1"
    assert rows[0]["min_positive_margin"] == pytest.approx(1.0)
    assert rows[0]["correct"] is True
