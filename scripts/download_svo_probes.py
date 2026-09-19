from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import io
import json
import random
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from urllib.parse import quote, unquote, urlparse, urlsplit, urlunsplit

from datasets import load_dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

SOURCE_DATASET = "MichiganNLP/svo_probes"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


class HostThrottle:
    """Small shared throttle so one host is not hit by many workers at once."""

    def __init__(self, delay: float = 0.0, jitter: float = 0.0) -> None:
        self.delay = max(0.0, float(delay))
        self.jitter = max(0.0, float(jitter))
        self._lock = threading.Lock()
        self._last_request: dict[str, float] = {}

    def wait(self, url: str) -> None:
        host = urlparse(url).netloc
        if not host or self.delay <= 0:
            if self.jitter > 0:
                time.sleep(random.uniform(0.0, self.jitter))
            return
        with self._lock:
            now = time.monotonic()
            next_allowed = self._last_request.get(host, 0.0) + self.delay
            sleep_for = max(0.0, next_allowed - now)
            self._last_request[host] = max(now, next_allowed)
        if sleep_for > 0:
            time.sleep(sleep_for)
        if self.jitter > 0:
            time.sleep(random.uniform(0.0, self.jitter))


def _image_records(dataset_name: str, split: str, cache_dir: str | None) -> dict[int, str]:
    records: dict[int, str] = {}
    dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    for row in dataset:
        records.setdefault(int(row["pos_image_id"]), str(row["pos_url"]))
        records.setdefault(int(row["neg_image_id"]), str(row["neg_url"]))
    return records


def _target_path(image_root: Path, image_id: int) -> Path:
    return image_root / f"{image_id:06d}.jpg"


def _quote_url(url: str) -> str:
    parts = urlsplit(str(url))
    path = quote(unquote(parts.path), safe="/:@")
    query = quote(unquote(parts.query), safe="=&?/:;+,%")
    return urlunsplit((parts.scheme, parts.netloc, path, query, parts.fragment))


def _candidate_urls(url: str, try_https_fallback: bool) -> list[str]:
    candidates = [str(url)]
    quoted = _quote_url(str(url))
    if quoted != candidates[0]:
        candidates.append(quoted)
    if try_https_fallback and str(url).startswith("http://"):
        https_url = "https://" + str(url)[len("http://") :]
        candidates.append(https_url)
        quoted_https = _quote_url(https_url)
        if quoted_https != https_url:
            candidates.append(quoted_https)
    seen = set()
    deduped = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _request_headers(url: str, user_agent: str, referer_mode: str) -> dict[str, str]:
    headers = {
        "User-Agent": user_agent,
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Upgrade-Insecure-Requests": "1",
    }
    if referer_mode == "origin":
        parsed = urlparse(url)
        if parsed.scheme and parsed.netloc:
            headers["Referer"] = f"{parsed.scheme}://{parsed.netloc}/"
    return headers


def _load_failed_manifest(root: Path) -> dict[int, str]:
    manifest_path = root / "download_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Cannot retry failed downloads; missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {int(row["image_id"]): str(row["url"]) for row in manifest.get("failed", [])}


def _download_one(
    item: tuple[int, str],
    image_root: Path,
    timeout: float,
    retries: int,
    user_agent: str,
    referer_mode: str,
    throttle: HostThrottle,
    try_https_fallback: bool,
) -> dict:
    image_id, url = item
    output_path = _target_path(image_root, image_id)
    if output_path.exists() and output_path.stat().st_size > 0:
        return {"image_id": image_id, "status": "exists", "path": str(output_path)}

    last_error = ""
    attempted_urls = _candidate_urls(url, try_https_fallback=try_https_fallback)
    for candidate_url in attempted_urls:
        headers = _request_headers(candidate_url, user_agent, referer_mode)
        for attempt in range(retries + 1):
            try:
                throttle.wait(candidate_url)
                request = urllib.request.Request(candidate_url, headers=headers)
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    payload = response.read()
                image = Image.open(io.BytesIO(payload)).convert("RGB")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = output_path.with_suffix(".tmp.jpg")
                image.save(tmp_path, format="JPEG", quality=95)
                tmp_path.replace(output_path)
                return {
                    "image_id": image_id,
                    "status": "downloaded",
                    "path": str(output_path),
                    "url": candidate_url,
                    "source_url": url,
                }
            except urllib.error.HTTPError as exc:
                last_error = repr(exc)
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                if retry_after:
                    try:
                        time.sleep(min(30.0, float(retry_after)))
                    except ValueError:
                        pass
                elif attempt < retries:
                    time.sleep(0.75 * (attempt + 1))
            except (
                OSError,
                http.client.IncompleteRead,
                urllib.error.URLError,
                TimeoutError,
                ValueError,
            ) as exc:
                last_error = repr(exc)
                if attempt < retries:
                    time.sleep(0.75 * (attempt + 1))

    return {
        "image_id": image_id,
        "status": "failed",
        "url": url,
        "attempted_urls": attempted_urls,
        "error": last_error,
    }


def download_svo_probes(
    root: str | Path,
    *,
    dataset_name: str = SOURCE_DATASET,
    split: str = "train",
    cache_dir: str | None = None,
    workers: int = 32,
    timeout: float = 15.0,
    retries: int = 1,
    max_images: int | None = None,
    retry_failed_from_manifest: bool = False,
    user_agent: str = DEFAULT_USER_AGENT,
    referer_mode: str = "origin",
    per_host_delay: float = 0.2,
    request_jitter: float = 0.05,
    try_https_fallback: bool = True,
) -> dict:
    root = Path(root).expanduser().resolve()
    image_root = root / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    if retry_failed_from_manifest:
        records = _load_failed_manifest(root)
    else:
        records = _image_records(dataset_name, split, cache_dir)
    items = sorted(records.items())
    if max_images is not None:
        items = items[: max(0, max_images)]

    manifest = {
        "source_dataset": dataset_name,
        "split": split,
        "root": str(root),
        "image_root": str(image_root),
        "n_requested": len(items),
        "n_downloaded": 0,
        "n_existing": 0,
        "n_failed": 0,
        "failed": [],
        "retry_failed_from_manifest": retry_failed_from_manifest,
        "user_agent": user_agent,
        "referer_mode": referer_mode,
        "workers": workers,
        "timeout": timeout,
        "retries": retries,
        "per_host_delay": per_host_delay,
        "request_jitter": request_jitter,
        "try_https_fallback": try_https_fallback,
    }
    throttle = HostThrottle(delay=per_host_delay, jitter=request_jitter)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(
                _download_one,
                item,
                image_root,
                timeout,
                retries,
                user_agent,
                referer_mode,
                throttle,
                try_https_fallback,
            )
            for item in items
        ]
        for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            status = result["status"]
            if status == "downloaded":
                manifest["n_downloaded"] += 1
            elif status == "exists":
                manifest["n_existing"] += 1
            else:
                manifest["n_failed"] += 1
                manifest["failed"].append(result)
            if idx % 250 == 0 or idx == len(futures):
                print(
                    f"[{idx}/{len(futures)}] downloaded={manifest['n_downloaded']} "
                    f"existing={manifest['n_existing']} failed={manifest['n_failed']}",
                    flush=True,
                )

    failure_errors = Counter(row.get("error", "unknown") for row in manifest["failed"])
    failure_domains = Counter(
        urlparse(str(row.get("url", ""))).netloc or "unknown" for row in manifest["failed"]
    )
    manifest["failure_error_counts"] = dict(failure_errors.most_common())
    manifest["failure_domain_counts"] = dict(failure_domains.most_common())
    manifest_name = (
        "download_manifest_retry_failed.json"
        if retry_failed_from_manifest
        else "download_manifest.json"
    )
    manifest_path = root / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Download SVO-Probes images.")
    parser.add_argument("--root", default="data/svo_probes")
    parser.add_argument("--dataset-name", default=SOURCE_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--retry-failed-from-manifest",
        action="store_true",
        help="Retry only URLs listed as failed in ROOT/download_manifest.json.",
    )
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument(
        "--referer-mode",
        choices=("origin", "none"),
        default="origin",
        help="Whether to send the URL origin as the Referer header.",
    )
    parser.add_argument("--per-host-delay", type=float, default=0.2)
    parser.add_argument("--request-jitter", type=float, default=0.05)
    parser.add_argument(
        "--no-https-fallback",
        action="store_true",
        help="Disable retrying http:// URLs as https:// URLs.",
    )
    args = parser.parse_args()
    download_svo_probes(
        args.root,
        dataset_name=args.dataset_name,
        split=args.split,
        cache_dir=args.cache_dir,
        workers=args.workers,
        timeout=args.timeout,
        retries=args.retries,
        max_images=args.max_images,
        retry_failed_from_manifest=args.retry_failed_from_manifest,
        user_agent=args.user_agent,
        referer_mode=args.referer_mode,
        per_host_delay=args.per_host_delay,
        request_jitter=args.request_jitter,
        try_https_fallback=not args.no_https_fallback,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
