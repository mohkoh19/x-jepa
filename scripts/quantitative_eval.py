from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class EvalSpec:
    alias: str
    experiment: str
    mode: str
    tags: tuple[str, ...]
    protocol: str
    backbone_frozen: bool
    trainable: tuple[str, ...]
    selection_split: str | None
    final_split: str
    image_resolution: int
    external_reference: bool = False
    matched_comparison: bool = True
    runner: str = "main"


EVALS: tuple[EvalSpec, ...] = (
    EvalSpec(
        "coco_zeroshot",
        "eval/coco_karpathy_zeroshot",
        "zero-shot",
        ("coco", "retrieval", "zeroshot"),
        "zeroshot",
        True,
        (),
        None,
        "test",
        224,
    ),
    EvalSpec(
        "coco_adapter_tune",
        "eval/coco_karpathy_adapter_tune",
        "retrieval_adapter_tune",
        ("coco", "retrieval", "adapter"),
        "retrieval_adapter_tune",
        True,
        ("image_retrieval_adapter", "text_retrieval_adapter", "logit_scale"),
        "val",
        "test",
        224,
    ),
    EvalSpec(
        "flickr30k_zeroshot",
        "eval/flickr30k_zeroshot",
        "zero-shot",
        ("flickr30k", "retrieval", "zeroshot"),
        "zeroshot",
        True,
        (),
        None,
        "test",
        224,
    ),
    EvalSpec(
        "flickr30k_adapter_tune",
        "eval/flickr30k_adapter_tune",
        "retrieval_adapter_tune",
        ("flickr30k", "retrieval", "adapter"),
        "retrieval_adapter_tune",
        True,
        ("image_retrieval_adapter", "text_retrieval_adapter", "logit_scale"),
        "val",
        "test",
        224,
    ),
    EvalSpec(
        "sugarcrepe_pp",
        "eval/sugarcrepe_pp",
        "zero-shot",
        ("sugarcrepe", "zeroshot"),
        "zeroshot",
        True,
        (),
        None,
        "sugarcrepe_pp",
        224,
    ),
    EvalSpec(
        "vsr",
        "eval/vsr",
        "zero-shot",
        ("vsr", "spatial", "compositional", "zeroshot"),
        "zeroshot",
        True,
        (),
        None,
        "vsr_test",
        224,
    ),
    EvalSpec(
        "svo_probes",
        "eval/svo_probes",
        "zero-shot",
        ("svo-probes", "verb", "compositional", "zeroshot"),
        "zeroshot",
        True,
        (),
        None,
        "svo_probes",
        224,
    ),
    EvalSpec(
        "nlvr2_token_interaction_probe",
        "eval/nlvr2_token_interaction_probe",
        "token_interaction_frozen_probe",
        ("nlvr2", "probe", "token-interaction"),
        "token_interaction_frozen_probe",
        True,
        ("token_interaction_transformer", "binary_classifier"),
        "dev",
        "test",
        224,
    ),
    EvalSpec(
        "nlvr2_global_raw_probe",
        "eval/nlvr2_global_raw_probe",
        "global_raw_frozen_probe",
        ("nlvr2", "probe", "global-raw"),
        "global_raw_frozen_probe",
        True,
        ("mlp_interaction_classifier",),
        "dev",
        "test",
        224,
    ),
)

EVAL_BY_ALIAS = {spec.alias: spec for spec in EVALS}
INTERNAL_ALIASES = (
    "coco_zeroshot",
    "flickr30k_zeroshot",
    "sugarcrepe_pp",
    "vsr",
    "svo_probes",
    "coco_adapter_tune",
    "flickr30k_adapter_tune",
    "nlvr2_token_interaction_probe",
    "nlvr2_global_raw_probe",
)
GROUPS: dict[str, tuple[str, ...]] = {
    "all": INTERNAL_ALIASES,
    "zero-shot": (
        "coco_zeroshot",
        "flickr30k_zeroshot",
        "sugarcrepe_pp",
        "vsr",
        "svo_probes",
    ),
    "zero_shot": (
        "coco_zeroshot",
        "flickr30k_zeroshot",
        "sugarcrepe_pp",
        "vsr",
        "svo_probes",
    ),
    "zeroshot": (
        "coco_zeroshot",
        "flickr30k_zeroshot",
        "sugarcrepe_pp",
        "vsr",
        "svo_probes",
    ),
    "coco": ("coco_zeroshot", "coco_adapter_tune"),
    "flickr30k": ("flickr30k_zeroshot", "flickr30k_adapter_tune"),
    "retrieval": (
        "coco_zeroshot",
        "flickr30k_zeroshot",
        "coco_adapter_tune",
        "flickr30k_adapter_tune",
    ),
    "adapter": (
        "coco_adapter_tune",
        "flickr30k_adapter_tune",
    ),
    "spatial": ("vsr",),
    "compositional": ("sugarcrepe_pp", "vsr", "svo_probes"),
    "alignment": (
        "coco_zeroshot",
        "flickr30k_zeroshot",
        "sugarcrepe_pp",
        "svo_probes",
        "vsr",
    ),
    "interaction": ("nlvr2_token_interaction_probe",),
    "diagnostic": ("nlvr2_global_raw_probe",),
    "probe": (
        "nlvr2_token_interaction_probe",
        "nlvr2_global_raw_probe",
    ),
    "paper_core": (
        "coco_zeroshot",
        "flickr30k_zeroshot",
        "sugarcrepe_pp",
        "svo_probes",
        "vsr",
        "coco_adapter_tune",
        "flickr30k_adapter_tune",
        "nlvr2_token_interaction_probe",
        "nlvr2_global_raw_probe",
    ),
    "debug_core": (
        "coco_zeroshot",
        "flickr30k_zeroshot",
        "sugarcrepe_pp",
        "svo_probes",
        "coco_adapter_tune",
        "nlvr2_token_interaction_probe",
    ),
}


@dataclass(frozen=True)
class ResolvedRun:
    run_dir: Path
    checkpoint_path: Path | None = None
    run_dirs: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if not self.run_dirs:
            object.__setattr__(self, "run_dirs", (self.run_dir,))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _split_csv(values: Iterable[str] | None) -> list[str]:
    tokens: list[str] = []
    for value in values or []:
        tokens.extend(part.strip() for part in value.split(",") if part.strip())
    return tokens


def expand_eval_selection(
    include: Iterable[str] | None, exclude: Iterable[str] | None
) -> list[EvalSpec]:
    include_tokens = _split_csv(include) or ["all"]
    exclude_tokens = _split_csv(exclude)

    def expand(tokens: Sequence[str]) -> list[str]:
        aliases: list[str] = []
        for token in tokens:
            if token in EVAL_BY_ALIAS:
                aliases.append(token)
            elif token in GROUPS:
                aliases.extend(GROUPS[token])
            else:
                valid = sorted({*EVAL_BY_ALIAS.keys(), *GROUPS.keys()})
                raise ValueError(
                    f"Unknown eval or group `{token}`. Valid choices: {', '.join(valid)}"
                )
        return aliases

    selected = []
    seen = set()
    for alias in expand(include_tokens):
        if alias not in seen:
            selected.append(alias)
            seen.add(alias)

    excluded = set(expand(exclude_tokens))
    return [EVAL_BY_ALIAS[alias] for alias in selected if alias not in excluded]


def _find_run_dir(path: Path) -> Path | None:
    """Return the Hydra run directory that contains ``path``, if any."""
    start = path.parent if path.is_file() else path
    for candidate in (start, *start.parents):
        if (candidate / ".hydra" / "config.yaml").exists():
            return candidate
    return None


def resolve_run_reference(reference: str) -> ResolvedRun:
    """Resolve a checkpoint file or Hydra run directory."""
    requested = Path(reference).expanduser()
    if requested.exists():
        resolved = requested.resolve()
        run_dir = _find_run_dir(resolved)
        if run_dir is None:
            if resolved.is_file() and resolved.suffix in {".ckpt", ".pth"}:
                # A released checkpoint: evaluation resolves its architecture
                # from configs/checkpoints/<name>.yaml.
                return ResolvedRun(run_dir=resolved.parent, checkpoint_path=resolved)
            raise FileNotFoundError(f"Could not find a parent Hydra run directory for {resolved}")
        checkpoint = (
            resolved if resolved.is_file() and resolved.suffix in {".ckpt", ".pth"} else None
        )
        return ResolvedRun(run_dir=run_dir, checkpoint_path=checkpoint, run_dirs=(run_dir,))

    raise FileNotFoundError(f"Could not resolve `{reference}` as a path.")


def best_checkpoint_file(run_dir: Path) -> Path:
    return run_dir / "checkpoint_selection" / "best_checkpoint.txt"


def read_selected_checkpoint(run_dir: Path) -> Path:
    path = best_checkpoint_file(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing selected checkpoint file: {path}")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"Selected checkpoint file is empty: {path}")
    return Path(value).expanduser().resolve()


def _hydra_list(values: Sequence[Path | str]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def selection_command(run_dir: Path, additional_run_dirs: Sequence[Path] = ()) -> list[str]:
    command = [sys.executable, "src/select_checkpoint.py", f"run_dir={run_dir}"]
    if additional_run_dirs:
        command.append(f"selector.additional_run_dirs={_hydra_list(additional_run_dirs)}")
    return command


def _selection_artifact_matches_run_dirs(run_dir: Path, run_dirs: Sequence[Path]) -> bool:
    if len(run_dirs) <= 1:
        return True
    selection_path = run_dir / "checkpoint_selection" / "selection.json"
    if not selection_path.exists():
        return False
    try:
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    recorded = payload.get("run_dirs")
    if not isinstance(recorded, list):
        return False
    expected = [str(Path(path).expanduser().resolve()) for path in run_dirs]
    recorded_paths = [str(Path(path).expanduser().resolve()) for path in recorded]
    return recorded_paths == expected


def parse_eval_overrides(values: Iterable[str] | None) -> dict[str, list[str]]:
    overrides: dict[str, list[str]] = {}
    for value in values or []:
        if ":" not in value:
            raise ValueError(f"Eval override `{value}` must use ALIAS:KEY=VALUE syntax.")
        alias, override = value.split(":", maxsplit=1)
        alias = alias.strip()
        override = override.strip()
        if alias not in EVAL_BY_ALIAS:
            raise ValueError(f"Unknown eval alias in override: `{alias}`")
        if not override:
            raise ValueError(f"Eval override for `{alias}` is empty.")
        overrides.setdefault(alias, []).append(override)
    return overrides


def build_eval_command(
    spec: EvalSpec,
    checkpoint_path: Path | str | None,
    output_dir: Path,
    python_executable: str | Path | None = None,
    global_overrides: Sequence[str] | None = None,
    eval_overrides: dict[str, list[str]] | None = None,
    fast_debug: bool = False,
) -> list[str]:
    del fast_debug
    executable = str(python_executable or sys.executable)
    ckpt_override = "null" if checkpoint_path is None else str(checkpoint_path)
    command = [
        executable,
        "src/eval.py",
        f"experiment={spec.experiment}",
        f"ckpt_path={ckpt_override}",
        f"hydra.run.dir={output_dir}",
    ]
    command.extend(global_overrides or [])
    command.extend((eval_overrides or {}).get(spec.alias, []))
    return command


def fast_debug_overrides(spec: EvalSpec) -> list[str]:
    overrides = [
        "++evaluation.fast_debug=true",
        "++trainer.devices=1",
    ]
    if spec.alias in {"coco_adapter_tune", "flickr30k_adapter_tune"}:
        overrides.extend(
            [
                "++trainer.min_epochs=1",
                "++trainer.max_epochs=1",
                "++trainer.check_val_every_n_epoch=1",
                "++trainer.limit_train_batches=2",
                "++trainer.limit_val_batches=2",
                "++trainer.limit_test_batches=2",
                "++data.batch_size=16",
                "++data.num_workers=0",
            ]
        )
    elif spec.alias in {"nlvr2_token_interaction_probe", "nlvr2_global_raw_probe"}:
        overrides.extend(
            [
                "++trainer.min_epochs=1",
                "++trainer.max_epochs=1",
                "++trainer.check_val_every_n_epoch=1",
                "++trainer.limit_train_batches=2",
                "++trainer.limit_val_batches=2",
                "++trainer.limit_test_batches=2",
                "++data.batch_size=16",
                "++data.num_workers=0",
                "++data.skip_missing_images=true",
                "++data.max_samples=128",
            ]
        )
    elif spec.alias in {"vsr", "svo_probes"}:
        overrides.extend(
            [
                "++trainer.limit_val_batches=2",
                "++data.batch_size=16",
                "++data.num_workers=0",
                "++data.max_samples=128",
            ]
        )
    else:
        overrides.extend(
            [
                "++trainer.limit_val_batches=2",
                "++trainer.limit_test_batches=2",
                "++data.batch_size=16",
                "++data.num_workers=0",
            ]
        )
    return overrides


def _summary_metadata(
    spec: EvalSpec,
    *,
    python_executable: str,
    eval_overrides: dict[str, list[str]],
    fast_debug: bool,
) -> dict:
    del python_executable, eval_overrides
    metadata = {
        "runner": spec.runner,
        "external_reference": spec.external_reference,
        "matched_comparison": spec.matched_comparison,
        "fast_debug": fast_debug,
    }
    return metadata


def _print_eval_registry() -> None:
    print("Available evals:")
    for spec in EVALS:
        print(f"  {spec.alias:22s} {spec.mode:9s} {spec.experiment}")
    print("\nGroups:")
    for name, aliases in GROUPS.items():
        print(f"  {name:22s} {', '.join(aliases)}")


def _run_subprocess(command: Sequence[str], cwd: Path) -> int:
    print(shlex.join(command), flush=True)
    return subprocess.run(command, cwd=cwd).returncode


def _read_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return {
        "metrics": payload.get("metrics"),
        "logged_metrics": payload.get("logged_metrics"),
    }


def run(args: argparse.Namespace) -> int:
    if args.list_evals:
        _print_eval_registry()
        return 0

    specs = expand_eval_selection(args.include, args.exclude)
    if not specs:
        raise ValueError("No evaluations selected.")

    repo_root = _repo_root()
    ckpt_path = getattr(args, "ckpt_path", None)
    output_dir_arg = getattr(args, "output_dir", None)
    if ckpt_path:
        resolved = (
            resolve_run_reference(args.run)
            if args.run
            else ResolvedRun(run_dir=Path(output_dir_arg or repo_root / "outputs").expanduser())
        )
        selected_checkpoint = Path(ckpt_path).expanduser()
        selection = {
            "command": None,
            "ran": False,
            "skipped": True,
        }
    else:
        resolved = resolve_run_reference(args.run)
        selected_checkpoint = ""
        selection = {
            "command": selection_command(resolved.run_dir, resolved.run_dirs[:-1]),
            "ran": False,
            "skipped": False,
        }
    run_dir = resolved.run_dir
    run_dirs = tuple(getattr(resolved, "run_dirs", (run_dir,)))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suite_dir = (
        Path(output_dir_arg).expanduser() / timestamp
        if output_dir_arg
        else run_dir / "quantitative_eval" / timestamp
    )
    eval_overrides = parse_eval_overrides(args.eval_override)
    if args.fast_debug:
        print("FAST DEBUG MODE: results are not scientifically meaningful.", flush=True)

    if not ckpt_path:
        if resolved.checkpoint_path is not None:
            selected_checkpoint = resolved.checkpoint_path
            selection["skipped"] = True
        elif args.skip_selection:
            selected_checkpoint = read_selected_checkpoint(run_dir)
            selection["skipped"] = True
        elif (
            best_checkpoint_file(run_dir).exists()
            and not args.force_selection
            and _selection_artifact_matches_run_dirs(run_dir, run_dirs)
        ):
            selected_checkpoint = read_selected_checkpoint(run_dir)
            selection["skipped"] = True
        elif args.dry_run:
            selected_checkpoint = "<selected-checkpoint-after-selection>"
        else:
            return_code = _run_subprocess(selection["command"], cwd=repo_root)
            selection["ran"] = True
            if return_code != 0:
                return return_code
            selected_checkpoint = read_selected_checkpoint(run_dir)

    results = []
    for spec in specs:
        output_dir = suite_dir / spec.alias
        python_executable = sys.executable
        debug_overrides = fast_debug_overrides(spec) if args.fast_debug else []
        command = build_eval_command(
            spec,
            selected_checkpoint,
            output_dir,
            python_executable=python_executable,
            global_overrides=[
                *debug_overrides,
                *(
                    [f"paths.data_dir={args.data_dir}"]
                    if getattr(args, "data_dir", None)
                    else []
                ),
                *args.global_override,
            ],
            eval_overrides=eval_overrides,
            fast_debug=bool(args.fast_debug),
        )
        metadata = _summary_metadata(
            spec,
            python_executable=str(python_executable),
            eval_overrides=eval_overrides,
            fast_debug=bool(args.fast_debug),
        )
        result = {
            "alias": spec.alias,
            "experiment": spec.experiment,
            "mode": spec.mode,
            "protocol": spec.protocol,
            "backbone_frozen": spec.backbone_frozen,
            "trainable": list(spec.trainable),
            "selection_split": spec.selection_split,
            "final_split": spec.final_split,
            "image_resolution": spec.image_resolution,
            **metadata,
            "output_dir": str(output_dir),
            "command": command,
            "returncode": None,
            "metrics_path": str(output_dir / "evaluation_metrics.json"),
            "metrics": None,
            "logged_metrics": None,
        }
        if args.dry_run:
            print(shlex.join(command))
        else:
            return_code = _run_subprocess(command, cwd=repo_root)
            result["returncode"] = return_code
            metrics = _read_metrics(output_dir / "evaluation_metrics.json")
            if metrics:
                result.update(metrics)
            if return_code != 0 and args.fail_fast:
                results.append(result)
                break
        results.append(result)

    summary = {
        "run_dir": str(run_dir),
        "run_dirs": [str(path) for path in run_dirs],
        "selected_checkpoint": None if selected_checkpoint is None else str(selected_checkpoint),
        "dry_run": bool(args.dry_run),
        "fast_debug": bool(args.fast_debug),
        "selection": {
            **selection,
            "command": selection["command"],
            "artifact": str(best_checkpoint_file(run_dir)) if selection["command"] else None,
        },
        "evaluations": results,
    }

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0

    suite_dir.mkdir(parents=True, exist_ok=True)
    (suite_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote summary to {suite_dir / 'summary.json'}")
    failed = [result for result in results if result["returncode"] not in (0, None)]
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select a pretraining checkpoint and run the quantitative evaluation suite."
    )
    parser.add_argument(
        "run",
        nargs="?",
        help="Checkpoint file or Hydra training run directory to evaluate.",
    )
    parser.add_argument(
        "--ckpt-path",
        "--ckpt_path",
        dest="ckpt_path",
        default=None,
        help="Explicit checkpoint path. Skips checkpoint selection and may be used without RUN.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output root for quantitative eval summaries. Useful with --ckpt-path.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Dataset root passed to every evaluation as `paths.data_dir`.",
    )
    parser.add_argument("--include", action="append", help="Comma-separated eval aliases/groups")
    parser.add_argument(
        "--evals",
        dest="include",
        action="append",
        help="Alias for --include.",
    )
    parser.add_argument("--exclude", action="append", help="Comma-separated eval aliases/groups")
    parser.add_argument("--list-evals", action="store_true")
    parser.add_argument("--force-selection", action="store_true")
    parser.add_argument("--skip-selection", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fast-debug",
        action="store_true",
        help="Append tiny Hydra overrides for smoke tests; results are not meaningful.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--override",
        dest="global_override",
        action="append",
        default=[],
        help="Hydra override appended to every evaluation.",
    )
    parser.add_argument(
        "--eval-override",
        action="append",
        default=[],
        help="Hydra override for one eval, using ALIAS:KEY=VALUE.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_evals:
        _print_eval_registry()
        return 0
    if not args.run and not args.ckpt_path:
        try:
            expand_eval_selection(args.include, args.exclude)
        except ValueError as exc:
            parser.error(str(exc))
        print(
            "error: the following arguments are required: run or --ckpt-path",
            file=sys.stderr,
        )
        return 2
    try:
        return run(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
