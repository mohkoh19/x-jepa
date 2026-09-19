#!/usr/bin/env python3
"""Convert the VL4M JSON manifest into WebDataset tar shards."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

JSON_CHUNK_SIZE = 1 << 20
DEFAULT_PROGRESS_EVERY = 100_000
DEFAULT_HEARTBEAT_SECONDS = 10.0
IMAGE_KEYS = {".jpg": "jpg", ".jpeg": "jpg", ".png": "png"}


@dataclass
class ShardInfo:
    name: str
    samples: int
    bytes: int


@dataclass
class BuildReport:
    version: int = 2
    format: str = "webdataset"
    source_json: str = ""
    root_dir: str = ""
    output_dir: str = ""
    source_sha256: str = ""
    expected_samples: int | None = None
    shard_pattern: str = ""
    maxcount: int = 0
    maxsize: int = 0
    samples: int = 0
    dropped_missing_image: int = 0
    dropped_invalid_record: int = 0
    dropped_decode_error: int = 0
    shards: list[ShardInfo] = field(default_factory=list)
    missing_examples: list[str] = field(default_factory=list)
    decode_error_examples: list[str] = field(default_factory=list)


class ProgressReporter:
    def __init__(
        self,
        every_rows: int = DEFAULT_PROGRESS_EVERY,
        heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
    ) -> None:
        self.every_rows = every_rows
        self.heartbeat_seconds = heartbeat_seconds
        self.start_time = time.monotonic()
        self.last_report_time = self.start_time
        self.last_report_rows = 0

    def update(self, rows: int, message: str | None = None, *, force: bool = False) -> None:
        now = time.monotonic()
        reached_rows = rows - self.last_report_rows >= self.every_rows
        reached_heartbeat = now - self.last_report_time >= self.heartbeat_seconds
        if not force and not reached_rows and not reached_heartbeat:
            return

        elapsed = max(now - self.start_time, 1e-9)
        suffix = f" | {message}" if message else ""
        print(
            f"[vl4m-wds] rows={rows:,} elapsed={elapsed / 60:.1f}m "
            f"rate={rows / elapsed:,.1f}/s{suffix}",
            flush=True,
        )
        self.last_report_time = now
        self.last_report_rows = rows


def iter_json_array(path: Path) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        started = False
        eof = False
        while True:
            if not eof and len(buffer) < JSON_CHUNK_SIZE // 2:
                chunk = handle.read(JSON_CHUNK_SIZE)
                if chunk == "":
                    eof = True
                else:
                    buffer += chunk

            if not started:
                buffer = buffer.lstrip()
                if not buffer and not eof:
                    continue
                if not buffer:
                    raise ValueError(f"Empty JSON file: {path}")
                if buffer[0] != "[":
                    raise ValueError(f"Expected top-level array in {path}")
                started = True
                buffer = buffer[1:]

            while True:
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if buffer[0] == "]":
                    return
                if buffer[0] == ",":
                    buffer = buffer[1:]
                    continue
                try:
                    obj, index = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    break
                if not isinstance(obj, dict):
                    raise ValueError(f"Expected object entries in array: {path}")
                yield obj
                buffer = buffer[index:]

            if eof:
                if buffer.strip() in {"", "]"}:
                    return
                raise ValueError(f"Unexpected trailing content in {path}")


def normalize_path(path: str) -> str:
    return str(path).strip().replace("\\", "/").lstrip("/")


def source_name(path: str) -> str:
    source = normalize_path(path).split("/", 1)[0].upper()
    return source or "VL4M"


def add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = 0
    tar.addfile(info, io.BytesIO(payload))


class ShardWriter:
    def __init__(self, output_dir: Path, pattern: str, maxcount: int, maxsize: int) -> None:
        self.output_dir = output_dir
        self.pattern = pattern
        self.maxcount = maxcount
        self.maxsize = maxsize
        self.shard_index = 0
        self.tar: tarfile.TarFile | None = None
        self.path: Path | None = None
        self.tmp_path: Path | None = None
        self.count = 0
        self.size = 0
        self.shards: list[ShardInfo] = []

    def _target_path(self) -> Path:
        return self.output_dir / (self.pattern % self.shard_index)

    def _open_next(self) -> None:
        self.path = self._target_path()
        self.tmp_path = self.path.with_name(f"{self.path.name}.tmp")
        self.tar = tarfile.open(self.tmp_path, "w")
        self.count = 0
        self.size = 0

    def _close_current(self) -> None:
        if self.tar is None or self.path is None or self.tmp_path is None:
            return
        self.tar.close()
        self.tmp_path.replace(self.path)
        self.shards.append(
            ShardInfo(name=self.path.name, samples=self.count, bytes=self.path.stat().st_size)
        )
        self.tar = None
        self.path = None
        self.tmp_path = None
        self.shard_index += 1

    def write(
        self, key: str, image_ext: str, image_bytes: bytes, metadata: dict[str, Any]
    ) -> None:
        json_bytes = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        added_size = len(image_bytes) + len(json_bytes)
        if (
            self.tar is None
            or self.count >= self.maxcount
            or (self.count > 0 and self.size + added_size > self.maxsize)
        ):
            self._close_current()
            self._open_next()

        assert self.tar is not None
        image_key = IMAGE_KEYS.get(image_ext.lower(), image_ext.lower().lstrip("."))
        add_bytes(self.tar, f"{key}.{image_key}", image_bytes)
        add_bytes(self.tar, f"{key}.json", json_bytes)
        self.count += 1
        self.size += added_size

    def close(self) -> None:
        self._close_current()

    def __enter__(self) -> ShardWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
            return
        if self.tar is not None:
            self.tar.close()
        if self.tmp_path is not None and self.tmp_path.exists():
            self.tmp_path.unlink()


def ensure_output_dir(output_dir: Path, pattern: str, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    glob_pattern = pattern.replace("%06d", "*").replace("%05d", "*").replace("%04d", "*")
    existing = sorted(output_dir.glob(glob_pattern))
    if existing and not overwrite:
        examples = ", ".join(path.name for path in existing[:3])
        raise FileExistsError(
            f"Output shards already exist in {output_dir}: {examples}. "
            "Pass --overwrite to remove matching shards first."
        )
    if overwrite:
        for path in existing:
            path.unlink()
        for path in output_dir.glob("*.tmp"):
            path.unlink()


def verify_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_webdataset(args: argparse.Namespace) -> BuildReport:
    input_json = Path(args.input_json).expanduser().resolve()
    root_dir = Path(args.root_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    maxsize = int(args.maxsize_gb * (1024**3))
    ensure_output_dir(output_dir, args.pattern, args.overwrite)

    report = BuildReport(
        source_json=str(input_json),
        root_dir=str(root_dir),
        output_dir=str(output_dir),
        source_sha256=sha256_file(input_json),
        expected_samples=getattr(args, "expected_samples", None),
        shard_pattern=args.pattern,
        maxcount=args.maxcount,
        maxsize=maxsize,
    )
    progress = ProgressReporter(args.progress_every)

    with ShardWriter(output_dir, args.pattern, args.maxcount, maxsize) as writer:
        for row_idx, row in enumerate(iter_json_array(input_json)):
            if args.limit is not None and row_idx >= args.limit:
                break

            image_path = row.get("image_path", row.get("image"))
            caption = row.get("caption", "")
            if not isinstance(image_path, str) or not image_path.strip():
                report.dropped_invalid_record += 1
                continue

            logical_path = normalize_path(image_path)
            resolved_path = Path(image_path).expanduser()
            if not resolved_path.is_absolute():
                resolved_path = root_dir / logical_path

            if not resolved_path.is_file():
                report.dropped_missing_image += 1
                if len(report.missing_examples) < 10:
                    report.missing_examples.append(logical_path)
                continue

            suffix = resolved_path.suffix.lower()
            if suffix not in IMAGE_KEYS:
                report.dropped_invalid_record += 1
                continue

            if args.verify_images:
                try:
                    verify_image(resolved_path)
                except Exception:
                    report.dropped_decode_error += 1
                    if len(report.decode_error_examples) < 10:
                        report.decode_error_examples.append(logical_path)
                    continue

            image_bytes = resolved_path.read_bytes()
            metadata = {
                "idx": row_idx,
                "image_path": logical_path,
                "caption": str(caption),
                "source": source_name(logical_path),
            }
            writer.write(f"{row_idx:09d}", suffix, image_bytes, metadata)
            report.samples += 1
            progress.update(row_idx + 1, f"written={report.samples:,}")

    report.shards = writer.shards
    expected_samples = getattr(args, "expected_samples", None)
    if expected_samples is not None and report.samples != int(expected_samples):
        raise ValueError(
            f"Dataset contract mismatch: expected {int(expected_samples):,} samples, "
            f"but wrote {report.samples:,}."
        )
    report_path = output_dir / "dataset.json"
    report_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    progress.update(report.samples, f"finished shards={len(report.shards):,}", force=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        required=True,
        help="Cleaned VL4M manifest, e.g. /data/4M/4M_cleaned.json.",
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Directory that relative `image_path` entries resolve against.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Shader directory for the WebDataset build, e.g. /data/4M_wds/train.",
    )
    parser.add_argument("--pattern", default="vl4m-%06d.tar")
    parser.add_argument("--maxcount", type=int, default=1000)
    parser.add_argument("--maxsize-gb", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--verify-images", action="store_true")
    parser.add_argument(
        "--expected-samples",
        type=int,
        default=None,
        help="Fail unless the cleaned dataset contains exactly this many samples.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    build_webdataset(parse_args())


if __name__ == "__main__":
    main()
