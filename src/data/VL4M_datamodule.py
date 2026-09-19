import inspect
import json
import logging
import math
import os
import warnings
from array import array
from hashlib import blake2b
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.base_datamodule import BaseDatamodule
from src.data.components.captioning import CaptionPreprocessingMixin

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="Truncated File Read", category=UserWarning)
warnings.filterwarnings("ignore", message="Corrupt EXIF data.*", category=UserWarning)


class BaseJsonCaptionDataset(CaptionPreprocessingMixin, Dataset):
    def __init__(self, max_words: int = 64) -> None:
        self.max_words = max_words


class JsonArrayIndex:
    """Byte-offset index for a top-level JSON array of objects."""

    def __init__(self, json_path: Path, chunk_size: int = 1 << 20) -> None:
        self.json_path = json_path
        self.chunk_size = chunk_size
        self.offsets = self._build_offsets()

    def _build_offsets(self) -> array:
        offsets = array("Q")
        in_string = False
        escape = False
        object_depth = 0
        object_start: int | None = None
        array_started = False
        byte_pos = 0

        with self.json_path.open("rb") as handle:
            while True:
                chunk = handle.read(self.chunk_size)
                if not chunk:
                    break

                for byte in chunk:
                    ch = chr(byte)

                    if in_string:
                        if escape:
                            escape = False
                        elif ch == "\\":
                            escape = True
                        elif ch == '"':
                            in_string = False
                    else:
                        if ch == '"':
                            in_string = True
                        elif ch == "[" and not array_started:
                            array_started = True
                        elif ch == "{":
                            if object_depth == 0:
                                object_start = byte_pos
                            object_depth += 1
                        elif ch == "}":
                            object_depth -= 1
                            if object_depth < 0:
                                raise ValueError(f"Malformed JSON array in {self.json_path}")
                            if object_depth == 0 and object_start is not None:
                                offsets.extend((object_start, byte_pos + 1))
                                object_start = None

                    byte_pos += 1

        if not array_started:
            raise ValueError(f"Expected top-level JSON array in {self.json_path}")
        if in_string or object_depth != 0:
            raise ValueError(f"Incomplete JSON content in {self.json_path}")

        return offsets

    def __len__(self) -> int:
        return len(self.offsets) // 2

    def __getitem__(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += len(self)
        offset_idx = idx * 2
        return self.offsets[offset_idx], self.offsets[offset_idx + 1]


class VL4MDataset(BaseJsonCaptionDataset):
    def __init__(
        self,
        json_path: str,
        root_dir: Optional[str] = None,
        image_roots: Optional[dict[str, str]] = None,
        transform: object = None,
        max_words: int = 64,
        dataset_name: str = "VL4M",
    ) -> None:
        super().__init__(max_words=max_words)
        self.json_path = Path(json_path).expanduser().resolve()
        self.root_dir = Path(root_dir).expanduser().resolve() if root_dir else None
        self.image_roots = {
            key: Path(value).expanduser().resolve() for key, value in (image_roots or {}).items()
        }
        self.transform = transform
        self.dataset_name = dataset_name
        self._file_handle = None
        self._file_handle_pid = None
        self.index = JsonArrayIndex(self.json_path)
        logger.info("Indexed %s with %s samples", self.json_path, f"{len(self.index):,}")

    def __len__(self) -> int:
        return len(self.index)

    def _get_handle(self):
        current_pid = os.getpid()
        if (
            self._file_handle is None
            or self._file_handle.closed
            or self._file_handle_pid != current_pid
        ):
            if self._file_handle is not None and not self._file_handle.closed:
                self._file_handle.close()
            self._file_handle = self.json_path.open("rb")
            self._file_handle_pid = current_pid
        return self._file_handle

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file_handle"] = None
        state["_file_handle_pid"] = None
        return state

    def _load_entry(self, idx: int) -> dict[str, Any]:
        start, end = self.index[idx]
        handle = self._get_handle()
        handle.seek(start)
        payload = handle.read(end - start)
        entry = json.loads(payload)
        if not isinstance(entry, dict):
            raise ValueError(f"Expected object entry at index {idx} in {self.json_path}")
        return entry

    def _resolve_image_path(self, raw_path: str) -> str:
        if raw_path.startswith("~"):
            candidate = Path(raw_path).expanduser()
            if candidate.is_absolute():
                return str(candidate)

        if os.path.isabs(raw_path):
            return raw_path

        normalized = raw_path.replace("\\", "/").lstrip("/")
        prefix, _, tail = normalized.partition("/")

        if prefix in self.image_roots:
            root = self.image_roots[prefix]
            return str(root / tail) if tail else str(root)

        if self.root_dir is not None:
            return str(self.root_dir / normalized)

        return normalized

    def __getitem__(self, idx: int):
        entry = self._load_entry(idx)
        image_value = entry.get("image_path", entry.get("image"))
        caption_value = entry.get("caption", "")

        if not isinstance(image_value, str) or not image_value.strip():
            raise ValueError(f"Missing image path for sample {idx} in {self.json_path}")

        if not isinstance(caption_value, str):
            caption_value = str(caption_value)

        caption = self.pre_caption(caption_value).strip()
        image_path = self._resolve_image_path(image_value)

        with Image.open(image_path) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)

        source_name = image_value.replace("\\", "/").split("/", 1)[0].upper()
        source_name = source_name if source_name else self.dataset_name

        return image, (caption, ""), -1, idx, source_name


class VL4MDatamodule(BaseDatamodule):
    def __init__(
        self,
        train_json: str,
        val_json: Optional[str] = None,
        root_dir: Optional[str] = None,
        image_roots: Optional[dict[str, str]] = None,
        batch_size: int = 16,
        collate_fn: object = None,
        train_transform: object = None,
        val_transform: object = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        max_words: int = 64,
        val_batch_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["collate_fn"], logger=False)

        self.collate_fn = collate_fn
        self.train_json = str(Path(train_json).expanduser().resolve())
        self.val_json = str(Path(val_json).expanduser().resolve()) if val_json else None

        self._init_transform()

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", None):
            self.train_set = VL4MDataset(
                json_path=self.train_json,
                root_dir=self.hparams.root_dir,
                image_roots=self.hparams.image_roots,
                transform=self.train_transform,
                max_words=self.hparams.max_words,
                dataset_name="VL4M",
            )
            if self.val_json is not None:
                self.val_set = VL4MDataset(
                    json_path=self.val_json,
                    root_dir=self.hparams.root_dir,
                    image_roots=self.hparams.image_roots,
                    transform=self.val_transform,
                    max_words=self.hparams.max_words,
                    dataset_name="VAL",
                )
            else:
                self.val_set = None

            logger.info(
                "Train set size: %s, Val set size: %s",
                f"{len(self.train_set):,}",
                f"{len(self.val_set):,}" if self.val_set is not None else "None",
            )

    def val_dataloader(self):
        if self.val_set is None:
            return None
        batch_size = self.hparams.val_batch_size or self.hparams.batch_size
        return self._to_dataloader(
            self.val_set,
            batch_size,
            train=False,
        )


class SizedWebDataset:
    """Small length-only placeholder for Lightning/datamodule introspection."""

    def __init__(self, size: int) -> None:
        self.size = int(size)

    def __len__(self) -> int:
        return self.size


class SizedDataLoader(DataLoader):
    """DataLoader with an explicit epoch length for iterable datasets."""

    def __init__(self, *args, length: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.length = int(length)

    def __len__(self) -> int:
        return self.length


class VL4MWebDatasetSampleDecoder(CaptionPreprocessingMixin):
    def __init__(
        self,
        transform: object = None,
        max_words: int = 64,
        dataset_name: str = "VL4M",
    ) -> None:
        self.transform = transform
        self.max_words = max_words
        self.dataset_name = dataset_name

    @staticmethod
    def _load_metadata(value: object) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            metadata = json.loads(value)
            if isinstance(metadata, dict):
                return metadata
        raise ValueError(f"Expected WebDataset JSON metadata, got {type(value).__name__}")

    @staticmethod
    def _load_image(value: object) -> Image.Image:
        if isinstance(value, Image.Image):
            if value.mode == "RGB":
                return value
            return value.convert("RGB")
        if isinstance(value, bytes):
            from io import BytesIO

            with Image.open(BytesIO(value)) as image:
                return image.convert("RGB")
        raise ValueError(f"Expected WebDataset image payload, got {type(value).__name__}")

    def __call__(self, sample: dict[str, Any]):
        image_value = sample.get("jpg", sample.get("jpeg", sample.get("png")))
        if image_value is None:
            raise ValueError(
                f"Missing image payload for WebDataset sample {sample.get('__key__')}"
            )

        metadata = self._load_metadata(sample.get("json"))
        image_path = str(metadata.get("image_path", metadata.get("image", "")))
        caption_value = metadata.get("caption", "")
        if not isinstance(caption_value, str):
            caption_value = str(caption_value)

        caption = self.pre_caption(caption_value).strip()
        image = self._load_image(image_value)
        if self.transform:
            image = self.transform(image)

        raw_idx = metadata.get("idx", metadata.get("index", sample.get("__key__", -1)))
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            idx = -1

        source_name = metadata.get("source")
        if not source_name:
            source_name = image_path.replace("\\", "/").split("/", 1)[0].upper()
        source_name = str(source_name or self.dataset_name)

        return image, (caption, ""), -1, idx, source_name


class VL4MWebDatasetSplitPredicate:
    def __init__(self, split: str, val_frac: float, seed: int | str | None = None) -> None:
        if split not in {"train", "val"}:
            raise ValueError("`split` must be either 'train' or 'val'")
        self.split = split
        self.val_frac = float(val_frac)
        self.seed = "" if seed in (None, "null") else str(seed)

    def __call__(self, sample: dict[str, Any]) -> bool:
        if self.val_frac <= 0.0:
            in_val = False
        elif self.val_frac >= 1.0:
            in_val = True
        else:
            sample_key = sample.get("__key__", "")
            sample_url = sample.get("__url__", "")
            key = f"{self.seed}:{sample_url}:{sample_key}".encode()
            digest = blake2b(key, digest_size=8).digest()
            value = int.from_bytes(digest, "big") / float(1 << 64)
            in_val = value < self.val_frac

        return in_val if self.split == "val" else not in_val


class VL4MWebDatasetDatamodule(BaseDatamodule):
    """VL4M datamodule backed by WebDataset shards.

    The emitted sample tuple intentionally matches ``VL4MDataset`` so all
    existing pretraining collators and models can be reused unchanged.
    """

    def __init__(
        self,
        train_shards: str | list[str],
        dataset_size: Optional[int] = None,
        metadata_path: Optional[str] = None,
        val_frac: float = 0.1,
        batch_size: int = 16,
        val_batch_size: Optional[int] = None,
        collate_fn: object = None,
        train_transform: object = None,
        val_transform: object = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        max_words: int = 64,
        shuffle_buffer: int = 10_000,
        shard_shuffle: int = 1_000,
        seed: int = 0,
        handler: str = "reraise",
    ) -> None:
        super().__init__()
        val_frac = self._validate_val_frac(val_frac)
        self.save_hyperparameters(ignore=["collate_fn"], logger=False)

        self.collate_fn = collate_fn
        self.train_shards = train_shards
        self.metadata_path = (
            str(Path(metadata_path).expanduser().resolve()) if metadata_path else None
        )
        self._init_transform()

        self.dataset_size = int(dataset_size) if dataset_size is not None else None
        self.train_size: int | None = None
        self.val_size: int | None = None
        self._shard_count: int | None = None

    @staticmethod
    def _validate_val_frac(value: float) -> float:
        val_frac = float(value)
        if not 0.0 <= val_frac <= 1.0:
            raise ValueError("`val_frac` must be a percentage in [0, 1].")
        return val_frac

    @staticmethod
    def _split_sizes(dataset_size: int, val_frac: float) -> tuple[int, int]:
        dataset_size = int(dataset_size)
        if dataset_size < 0:
            raise ValueError("`dataset_size` must be non-negative.")
        if dataset_size == 0 or val_frac <= 0.0:
            return dataset_size, 0
        if val_frac >= 1.0:
            return 0, dataset_size

        val_size = int(round(dataset_size * val_frac))
        val_size = max(1, min(dataset_size - 1, val_size))
        return dataset_size - val_size, val_size

    def _load_metadata(self) -> None:
        if self.metadata_path is None:
            return

        path = Path(self.metadata_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing VL4M WebDataset metadata: {path}")
        metadata = json.loads(path.read_text(encoding="utf-8"))

        if self.dataset_size is None:
            samples = metadata.get("samples", metadata.get("dataset_size"))
            if samples is None:
                raise ValueError(f"Missing `samples` in WebDataset metadata: {path}")
            self.dataset_size = int(samples)

        shards = metadata.get("shards")
        if isinstance(shards, list):
            self._shard_count = len(shards)

    def setup(self, stage: Optional[str] = None):
        if stage not in ("fit", None):
            return

        self._load_metadata()
        if self.dataset_size is None:
            raise ValueError(
                "`dataset_size` is required for VL4MWebDatasetDatamodule unless "
                "`metadata_path` points to a builder report with a `samples` field."
            )

        self.train_size, self.val_size = self._split_sizes(
            self.dataset_size, self.hparams.val_frac
        )
        self.train_set = SizedWebDataset(self.train_size)
        self.val_set = SizedWebDataset(self.val_size) if self.val_size > 0 else None

        logger.info(
            "VL4M WebDataset size: %s samples across %s shards; train=%s, val=%s " "(val_frac=%s)",
            f"{self.dataset_size:,}",
            f"{self._shard_count:,}" if self._shard_count is not None else "unknown",
            f"{self.train_size:,}",
            f"{self.val_size:,}",
            self.hparams.val_frac,
        )

    def _world_info(self) -> tuple[int, int]:
        world_size = 1
        global_rank = 0
        if hasattr(self, "trainer") and self.trainer is not None:
            world_size = int(getattr(self.trainer, "world_size", 1))
            global_rank = int(getattr(self.trainer, "global_rank", 0))
        return max(1, world_size), max(0, global_rank)

    def _handler(self, wds):
        name = str(self.hparams.handler)
        if name == "warn_and_continue":
            return wds.warn_and_continue
        if name == "ignore_and_continue":
            return wds.ignore_and_continue
        if name == "reraise":
            return wds.reraise_exception
        raise ValueError(
            "`handler` must be one of: reraise, warn_and_continue, ignore_and_continue"
        )

    def _make_split_dataset(
        self,
        *,
        split: str,
        transform: object,
        shuffle: bool,
        samples_per_rank: int | None = None,
    ):
        try:
            import webdataset as wds
        except ImportError as exc:
            raise ImportError(
                "VL4MWebDatasetDatamodule requires the `webdataset` package. "
                "Install project dependencies after the pyproject update."
            ) from exc

        handler = self._handler(wds)
        decoder = VL4MWebDatasetSampleDecoder(
            transform=transform,
            max_words=self.hparams.max_words,
            dataset_name="VL4M" if split == "train" else "VAL",
        )
        seed = self.hparams.seed
        seed = None if seed in (None, "null") else int(seed)

        wds_kwargs = {
            "shardshuffle": int(self.hparams.shard_shuffle) if shuffle else False,
            "nodesplitter": wds.split_by_node,
            "handler": handler,
            "seed": seed,
        }
        if "workersplitter" in inspect.signature(wds.WebDataset).parameters:
            wds_kwargs["workersplitter"] = wds.split_by_worker
        else:
            wds_kwargs["splitter"] = wds.split_by_worker

        dataset = wds.WebDataset(self.train_shards, **wds_kwargs)
        dataset = dataset.select(
            VL4MWebDatasetSplitPredicate(
                split=split,
                val_frac=self.hparams.val_frac,
                seed=self.hparams.seed,
            )
        )
        if shuffle:
            dataset = dataset.shuffle(int(self.hparams.shuffle_buffer), handler=handler)
        dataset = dataset.decode("pilrgb", handler=handler)
        dataset = dataset.map(decoder, handler=handler)
        if samples_per_rank is not None:
            dataset = dataset.with_epoch(samples_per_rank)
        return dataset

    def train_dataloader(self):
        if self.train_size is None:
            self._load_metadata()
            if self.dataset_size is not None:
                self.train_size, self.val_size = self._split_sizes(
                    self.dataset_size, self.hparams.val_frac
                )
        if self.train_size is None:
            raise ValueError("VL4M WebDataset setup did not resolve dataset_size.")

        batch_size = self.hparams.batch_size
        world_size, _ = self._world_info()
        samples_per_rank = (int(self.train_size) // world_size // batch_size) * batch_size
        if samples_per_rank <= 0:
            raise ValueError(
                "Resolved zero VL4M WebDataset samples per rank. "
                f"train_size={self.train_size}, world_size={world_size}, "
                f"batch_size={batch_size}"
            )

        dataset = self._make_split_dataset(
            split="train",
            transform=self.train_transform,
            shuffle=True,
            samples_per_rank=samples_per_rank,
        )

        loader_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "collate_fn": self.collate_fn,
            "num_workers": self.hparams.num_workers,
            "pin_memory": self.hparams.pin_memory,
            "drop_last": True,
            "length": samples_per_rank // batch_size,
        }
        if self.hparams.num_workers > 0:
            loader_kwargs.update(
                persistent_workers=True,
                prefetch_factor=2,
            )

        return SizedDataLoader(
            **loader_kwargs,
        )

    def val_dataloader(self):
        if self.val_set is None:
            return None
        if self.val_size is None:
            raise ValueError("VL4M WebDataset setup did not resolve validation size.")

        batch_size = self.hparams.val_batch_size or self.hparams.batch_size
        world_size, _ = self._world_info()
        samples_per_rank = int(math.ceil(self.val_size / world_size))
        length = int(math.ceil(samples_per_rank / batch_size))

        dataset = self._make_split_dataset(
            split="val",
            transform=self.val_transform,
            shuffle=False,
        )

        loader_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "collate_fn": self.collate_fn,
            "num_workers": self.hparams.num_workers,
            "pin_memory": self.hparams.pin_memory,
            "drop_last": False,
            "length": length,
        }
        if self.hparams.num_workers > 0:
            loader_kwargs.update(
                persistent_workers=True,
                prefetch_factor=2,
            )

        return SizedDataLoader(**loader_kwargs)
